import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ----------------------------------------------------
# CONFIG + TOKEN
# ----------------------------------------------------
st.set_page_config(page_title="Transcript Q&A Chatbot", layout="wide")
st.title("Transcript Question Answering Chatbot")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.warning("HuggingFace API token missing.")
    st.info("Add HUGGINGFACEHUB_API_TOKEN in Space Secrets.")
    st.stop()

# ----------------------------------------------------
# UI INPUTS
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])
question = st.text_input("Ask a question:", value="Can you summarize this transcript?")

# ----------------------------------------------------
# RAG PIPELINE
# ----------------------------------------------------
@st.cache_resource(show_spinner=True)
def setup_rag(transcript_text):

    # ---- Text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([transcript_text])

    # ---- Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---- LLM
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.2,
            max_new_tokens=512,
        )
    )

    # ---- Prompt
    prompt = PromptTemplate(
        template = """
You are a helpful assistant.
Answer ONLY from the transcript context below.
If the answer cannot be found, reply with "I don't know".
TRANSCRIPT:
{context}
QUESTION:
{question}
""",
        input_variables=["context", "question"]
    )

    # ---- Format docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    chain = parallel_chain | prompt | llm | parser

    return chain


# ----------------------------------------------------
# MAIN FLOW
# ----------------------------------------------------
if uploaded_file:

    transcript_text = uploaded_file.read().decode("utf-8")

    with st.spinner("Building knowledge base..."):
        try:
            rag_chain = setup_rag(transcript_text)
        except Exception as err:
            st.error(f"Failed to build pipeline: {err}")
            st.stop()

    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            try:
                reply = rag_chain.invoke(question)
                st.subheader("Answer")
                st.write(reply)
            except Exception as err:
                st.error(f"Error generating response: {err}")

else:
    st.info("Upload a transcript (.txt file) to begin.")
