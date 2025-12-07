YouTube Transcript Q&A Chatbot (RAG Application)

Live Demo (Hugging Face Space):
https://huggingface.co/spaces/Mohan312/Youtube_Chatbot

Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload any transcript file (for example, a YouTube video transcript or document text) and ask natural-language questions about its content.

The system processes the uploaded transcript, splits the text into manageable chunks, creates semantic embeddings, stores them in a FAISS vector database, and retrieves the most relevant context for each user question. Answers are then generated using a Hugging Face large language model, ensuring that responses are grounded strictly in the provided transcript.

Key Features

Upload any transcript file (.txt) and query its content.

Full RAG pipeline implemented with LangChain.

Vector search with FAISS for fast and accurate retrieval.

Sentence-Transformers used for embedding generation.

Hugging Face LLM used for grounded answer generation.

Streamlit-based user interface.

Public cloud deployment on Hugging Face Spaces.

Secure API key management using environment variables.

System Architecture
Transcript Upload
        ↓
Text Chunking (LangChain)
        ↓
Embeddings (Sentence-Transformers)
        ↓
FAISS Vector Index
        ↓
Retriever
        ↓
Prompt + LLM (Hugging Face Zephyr)
        ↓
Answer Generation

Tech Stack

Python

Streamlit (Web UI)

LangChain (RAG orchestration)

FAISS (Vector database)

Sentence-Transformers (Text embeddings)

Hugging Face Hub (LLM inference)

Hugging Face Spaces (Cloud deployment)

Project Structure
YouTube_Chatbot/

├── app.py                 # Streamlit chatbot application
├── requirements.txt       # Python dependencies
├── rag_using_langchain.ipynb
│                          # Development notebook
└── README.md              # Project documentation

How to Use
Online (Recommended)

Open the live application:
https://huggingface.co/spaces/Mohan312/Youtube_Chatbot

Upload a transcript file (.txt).

Enter a question related to the transcript.

Click Ask to receive an answer.

Run Locally
pip install -r requirements.txt
streamlit run app.py


Then open:

http://localhost:8501


Upload a transcript file and start asking questions.

Creating a Transcript File

Transcripts can be created using the YouTube Transcript API locally or in Google Colab:

from youtube_transcript_api import YouTubeTranscriptApi

video_id = "YOUR_VIDEO_ID"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en"])
text = " ".join(chunk.text for chunk in transcript)

with open("video_transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)


Upload the generated .txt file into the web app.

Security

No API keys are hard-coded into the repository.

Secrets are managed using environment variables (Hugging Face Space Secrets).

Why This Project Matters

This project demonstrates:

Practical implementation of Retrieval-Augmented Generation systems.

Dynamic document ingestion pipelines.

Use of vector search for contextual retrieval.

LLM integration for grounded question answering.

Cloud deployment and model inference workflows.

Secure credential handling.

These elements represent real-world production patterns used in modern document-based AI systems.

Author

Mohan Kumar

GitHub: https://github.com/Mohankumar3124

Hugging Face: https://huggingface.co/Mohan312

Live Demo

https://huggingface.co/spaces/Mohan312/Youtube_Chatbot
