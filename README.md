# Topic Modeling on News Articles

## Overview
This project applies **topic modeling** techniques to analyze and extract latent themes from a dataset of news articles. By leveraging **Latent Dirichlet Allocation (LDA)** with **Count Vectorizer** and **TF-IDF**, we identified key topics within the dataset, categorized articles, and gained insights into the underlying thematic structure of the corpus.

## Dataset
The dataset consists of **2,200 news articles** spanning the following categories:
- Entertainment
- Business
- Technology
- Politics
- Sports

### Data Format:
Each record in the dataset includes:
- `Title`: Headline of the news article.
- `Description`: Brief content of the article.
- `Category`: Predefined category (used for validation).

## Methodology
### 1. Data Preprocessing
- Removed stop words, punctuation, and special characters.
- Tokenized and lemmatized text.
- Prepared feature vectors using:
  - **Count Vectorizer**: Word frequency representation.
  - **TF-IDF Vectorizer**: Weighted representation based on word importance.

### 2. Topic Modeling
- Applied **Latent Dirichlet Allocation (LDA)** using `sklearn` and `gensim`.
- Experimented with two feature extraction methods:
  - **Count Vectorizer**.
  - **TF-IDF Vectorizer**.

### 3. Evaluation
- Measured **coherence scores** to evaluate topic quality.
- Reviewed **top keywords** for interpretability.

## Results
### Identified Topics
Below are sample topics extracted from the dataset:

#### Topic 1: Politics
- Keywords: government, policy, election, party, law

#### Topic 2: Sports
- Keywords: match, team, win, player, championship

#### Topic 3: Technology
- Keywords: innovation, software, company, device, AI

#### Topic 4: Business
- Keywords: market, growth, investment, profit, economy

#### Topic 5: Entertainment
- Keywords: movie, actor, music, festival, award

### Sample Article Classification
- **Title**: "Tech Giants Invest in AI Startups"
  - **Category**: Technology

- **Title**: "Championship Game Ends in Dramatic Finish"
  - **Category**: Sports

## Project Files
- `data/`: Contains the raw and preprocessed datasets.
- `notebooks/`: Includes Jupyter notebooks for preprocessing, modeling, and evaluation.
- `src/`: Python scripts for modular processing and modeling.
- `results/`: Outputs such as topic keywords and classified articles.

## Requirements
Install the necessary dependencies using:
```bash
pip install -r requirements.txt
