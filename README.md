# 📈 Finance RAG: Autonomous Trading Research Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)

### 🧠 The "Brain" of an Autonomous Trading System

**Finance RAG** is a local, privacy-focused analysis tool designed to ingest complex financial documents (SEC 10-K filings, IPO reports, Earnings transcripts) and provide evidence-based answers. 

This project serves as **Module 1** of a larger **Autonomous Trading Bot** ecosystem. Before executing trades, the system uses this engine to understand the *fundamental* risks and sentiment hidden in unstructured text, complementing technical chart analysis.

---

## 🚀 Features

* **Local Privacy:** Runs entirely offline using **Ollama (Llama 3)**. No data leaves your machine.
* **RAG Architecture:** Uses Retrieval-Augmented Generation to minimize hallucinations.
* **Smart Ingestion:** Processes PDF documents using `RecursiveCharacterTextSplitter` to maintain financial context.
* **Vector Search:** Utilizes **ChromaDB** and **Nomic Embeddings** for semantic similarity search.
* **Source Citation:** The AI provides specific citations and excerpts for every answer to ensure trust.
* **Chat Interface:** A clean, WhatsApp-style UI built with **Streamlit**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **LLM Orchestration:** LangChain
* **Local LLM:** Ollama (Llama 3)
* **Vector Database:** ChromaDB
* **Embeddings:** Nomic-embed-text
* **Frontend:** Streamlit

---

## ⚙️ Installation & Setup

### 1. Prerequisites
You must have [Ollama](https://ollama.com/) installed on your system.
Once installed, pull the required models via your terminal:

```bash
ollama pull llama3
ollama pull nomic-embed-text
