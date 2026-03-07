\# Semantic Search System



AI/ML Engineer Assignment – Trademarkia



\## Overview

This project implements a lightweight semantic search system using the 20 Newsgroups dataset.



\## Features

\- Document embeddings using Sentence Transformers

\- Semantic search using cosine similarity

\- Custom semantic cache (no Redis or external cache)

\- FastAPI API service



\## API Endpoints



\### POST /query

Accepts a natural language query and returns the most semantically similar document.



Example request:



{

&nbsp;"query": "space shuttle launch"

}



\### GET /cache/stats

Returns cache statistics.



\### DELETE /cache

Clears the semantic cache.



\## Technologies Used

\- Python

\- FastAPI

\- Sentence Transformers

\- Scikit-learn

\- NumPy



\## Running the Project



Create virtual environment:



python -m venv venv



Activate environment:



venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt



Run the API:



uvicorn app.main:app --reload

