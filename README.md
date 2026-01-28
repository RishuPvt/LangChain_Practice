LangChain + Google Gemini AI Projects

This repository contains practical implementations and experiments using LangChain with Google Gemini models.
The goal of this project is to understand prompt engineering, chaining, conversational context, Streamlit apps, and embeddings, while building real, usable mini-applications.

📌 Project Overview

This repository includes:

A Kids Story Generator built with Streamlit

A CLI-based conversational chatbot with manual memory handling

Embedding model experiments for semantic understanding

Core LangChain concepts such as PromptTemplate, chains, and LLM invocation

✨ Features
1. Kids Story Generator (Streamlit)

A web-based application that generates creative children’s stories based on:

Child’s age

Gender

Story genre

Story length

Highlights

Uses PromptTemplate for structured prompting

Chains prompt → Gemini model using LangChain

Produces age-appropriate, engaging stories with positive morals

2. Conversational Chat (Command Line)

A terminal-based chatbot that:

Stores previous conversation manually

Explicitly passes chat history to the model

Demonstrates why LLMs do not remember context automatically

Purpose

Understand conversation memory limitations

Learn how context must be included in prompts

3. Embeddings Experiments

Basic experimentation with embedding models to understand:

Text vector representation

Semantic similarity

Foundations for retrieval-based AI systems (RAG)
