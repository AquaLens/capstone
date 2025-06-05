# AquaLens
Capstone Project for AquaLens

This Github Repository will contain all the coding and information of our Captstone Project website. 
It will also contain the data abehind our visualisations.

=====================================
Website Intervention - aqualens.info
=====================================

The website files are all located in the 'docs' folder. Each page's code is located in its own folder, with its respective name. To launch the website locally, launch a live server with the 'docs' folder as the root.

=====================================
AquaLens' AI Agent - Froggy
=====================================

Froggy is an app built and hosted on huggingface.co. The code is located in a separate folder called 'froggy-backend'.

A Flask-based API that provides computational search across water quality projects 
and datasets. The system combines:
- Project/initiative data from Sanity CMS/API
- Multiple CSV datasets with water quality metrics
- AI-powered response generation using Together AI
- Conversation history management
- Hybrid search (projects + data) capabilities

Main Features:
- Multi-dataset search with TF-IDF vectorization
- Query type detection (projects, data, hybrid, catalog)
- Country-specific data filtering
- Time-series data handling
- Session-based conversation management
