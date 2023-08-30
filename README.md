# LSA Proof of Concept ALT

Welcome to the **LSA-Proof-Of-Concept-ALT** repository! This project showcases a course recommendation system built using Latent Semantic Analysis (LSA). The system utilizes Python for its implementation and employs Flask to provide an interactive interface. The primary goal of this project is to demonstrate how course syllabi can be leveraged to identify and recommend courses that share similar content through the application of LSA.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [System Components](#system-components)
- [How It Works](#how-it-works)

## Overview

The **LSA-Proof-Of-Concept-ALT** repository contains the codebase for a course recommendation system. By utilizing Latent Semantic Analysis, course syllabi are transformed into numerical vectors, allowing for a quantitative measure of their semantic content. This vector representation enables the calculation of cosine similarity between course syllabi, which forms the basis for identifying courses with similar content.

## Installation

To set up the course recommendation system on your local machine, follow these steps:

1. Clone the repository:
git clone https://github.com/your-username/LSA-Proof-Of-Concept-ALT.git


2. Navigate to the project directory:
cd LSA-Proof-Of-Concept-ALT


3. Create a virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate


4. Install the required dependencies:
pip install -r requirements.txt
import re
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk import FreqDist, bigrams
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import Normalizer
import base64
from matplotlib.figure import Figure


## Usage

1. Run the Flask application: python app.py

2. Access the system by opening a web browser and navigating to `http://localhost:5000`.

3. Enter a course code (must be of the form "XXXX 111", e.g., "DATA 101"), the year you want the recommend course to be in (1-4 or 0 for any year), and whether to use TFIDF weighting or not (you may enter "y" or "n", generally TFIDF weighting improves text similarity)

4. Click "Submit" to print courses and their similarity scores as decided through the Cosine similarity.

## System Components

The **LSA-Proof-Of-Concept-ALT** repository consists of the following components:

- `app.py`: This is the main Flask application file that handles modelbuilding, user interactions, and displays the web interface.
- `data/`: This directory contains all data used to train the model. UBCO_Course_Calendar.csv is what is actually fed into the model. Other files are course syllabi that were cleaned and put into this CSV.
- `templates/`: This directory contains HTML templates for rendering the web interface.

## How It Works

1. **Text Preprocessing**: Course syllabi are preprocessed to remove stop words, punctuation, and perform stemming to normalize the text data.

2. **Document Vectorization**: Using the LSA technique, the preprocessed syllabi are transformed into numerical vectors that capture the latent semantic information of the text.

3. **Cosine Similarity**: Cosine similarity is calculated between the vectorized syllabi to determine their semantic similarity. Higher cosine similarity values indicate greater content similarity.

4. **Recommendation**: Given a user's input syllabus, the system identifies the most similar courses based on cosine similarity and presents them as recommendations.














