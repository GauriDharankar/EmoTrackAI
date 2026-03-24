# EmoTrack AI
**NLP-based Emotion Detection System with Streamlit Interface**

## Overview
EmoTrack AI is a machine learning-based system designed to detect user emotions from social media text.
It uses NLP techniques to process textual input and classify it into emotional categories.
The project also includes a basic Streamlit interface, allowing users to interact with the model in real time.

## Features
- Emotion detection from text input
- Text preprocessing (cleaning, tokenization, stopword removal)
- Feature extraction using TF-IDF
- Machine learning-based classification
- Simple Streamlit web interface for user interaction
- Multi-mode system:
  - Personal mode
  - Group analysis mode
  - Clinician mode

## System Workflow
- Input: Social media text
- Preprocessing:
  - Cleaning and normalization
  - Tokenization
  - Stopword removal
- Feature Extraction: TF-IDF vectorization
- Model: Trained classification model for emotion prediction
- Interface: Streamlit UI takes input and displays predicted emotion
- Output: Predicted emotion label

## Tech Stack
Language: Python
Libraries: Scikit-learn, Pandas
Framework: Streamlit
Techniques: TF-IDF, Text Preprocessing

## Evaluation
Model evaluated using:
- Accuracy: 0.83
- Weighted F1-score: 0.83

### Class-wise Performance:
- Anxiety → F1-score: 0.97  
- Depression → F1-score: 0.94  
- Normal → F1-score: 0.78  
- Stress → F1-score: 0.75  

## Datasets

The model was trained using a combination of multiple datasets, which were cleaned, merged, and standardized into a unified format.

### Datasets Used:
- **Mental Health Dataset**  
  - Primary dataset containing labeled text for emotions such as stress, anxiety, depression, and normal  

- **GoEmotions Dataset**  
  - Fine-grained emotion dataset used to improve diversity in emotional expressions  

- **Sentiment140 Dataset**  
  - Large-scale Twitter dataset used to enhance generalization on social media text  

### Final Dataset Summary:
- **Total Samples:** ~300,000  

### Label Distribution:
- Stress → 100,000  
- Normal → 100,000  
- Depression → 50,000  
- Anxiety → 50,000  

### Data Processing:
- Combined multiple datasets into a single unified dataset  
- Standardized text format and labels across datasets  
- Applied preprocessing (cleaning, tokenization, normalization)  
- Mapped labels into core emotion categories  

> The dataset was structured to maintain balanced representation across major emotion classes and improve model generalization.

## Current Status
Completed — NLP-based emotion detection pipeline implemented with a basic Streamlit interface

## Future Improvements
- Use transformer-based models (BERT, etc.) for improved accuracy
- Enhance Streamlit UI with better visualization
- Enable real-time emotion analysis from streaming data
- Add user-level insights for clinician and group modes

## Use Cases
- Mental health awareness
- Social media sentiment analysis
- Research and behavioral studies
- Support tools for clinicians

## How to Run

### Clone the repository
git clone https://github.com/your-username/emotrack-ai.git

### Navigate to project folder
cd emotrack-ai

### Install dependencies
pip install -r requirements.txt

### Run the Streamlit app
streamlit run app.py

## Author
**Gauri Dharankar**
Aspiring Machine Learning Engineer
