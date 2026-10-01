# AI-Based Disaster Tweet Classification and Monitoring Dashboard

## Project Overview

The AI-Based Disaster Tweet Classification and Monitoring Dashboard is a machine learning project that identifies disaster-related tweets using Natural Language Processing (NLP) and Machine Learning techniques.

During natural disasters, social media platforms such as Twitter (now X) can contain useful information about emergencies, affected areas, and requests for assistance. Manually analyzing large volumes of tweets can be difficult and time-consuming.

This project uses a machine learning model to classify tweets into disaster-related and non-disaster-related categories. An interactive dashboard developed using Streamlit displays the classification results and provides an overview of the dataset.

## Objectives

* To classify tweets as disaster-related or non-disaster-related.
* To apply NLP techniques for text preprocessing.
* To develop and train a machine learning model.
* To analyze disaster-related information from tweets.
* To visualize classification results through an interactive dashboard.
* To identify potentially critical tweets and assistance-related information.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* NLTK
* Streamlit
* Matplotlib
* Git and GitHub

## Project Structure

```text
Disaster-Tweet-Classification/
│
├── app.py
├── dataset/
│   └── train.csv
├── models/
├── pages/
├── utils/
│   └── preprocessing.py
├── explore_data.py
├── train_model.py
└── README.md
```

## Dataset

The project uses the Disaster Tweets dataset, which contains 7,613 tweets labeled as disaster-related or non-disaster-related.

Dataset distribution:

* Total tweets: 7,613
* Disaster-related tweets: 3,271
* Non-disaster-related tweets: 4,342

The dataset contains tweet text and corresponding classification labels.

* Label 1: Disaster-related tweet
* Label 0: Non-disaster-related tweet

## Methodology

### 1. Data Collection

The project uses a labeled dataset containing disaster-related and non-disaster-related tweets.

### 2. Data Preprocessing

Natural Language Processing techniques are used to clean and prepare the tweet text.

The preprocessing steps include:

* Converting text to lowercase.
* Removing unnecessary characters and punctuation.
* Removing stop words.
* Applying stemming.

### 3. Exploratory Data Analysis

The dataset is analyzed to understand the distribution of tweets and identify patterns in the data.

### 4. Model Training

A machine learning model is trained using the preprocessed tweet data. The model learns patterns from the training dataset to classify tweets.

### 5. Classification

The trained model predicts whether a tweet is disaster-related or non-disaster-related.

### 6. Dashboard Development

An interactive dashboard is developed using Streamlit to display the classification results and dataset statistics.

## Model Performance

The initial machine learning model achieved an accuracy of approximately 79.25% on the test dataset.

Model performance may vary depending on the training and testing data.

## Dashboard Features

* Displays the total number of tweets analyzed.
* Shows the number of disaster-related tweets.
* Displays classification results.
* Shows model confidence scores.
* Provides priority-based filtering.
* Displays assistance-related information.
* Includes interactive filters for exploring the results.

The current dashboard analyzes the tweets available in the provided dataset. It does not provide real-time social media monitoring.

## Installation and Setup

### Prerequisites

* Python 3
* Git
* Visual Studio Code (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/Elakyaa-07/Disaster-Tweet-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Disaster-Tweet-Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install the Required Libraries

```bash
pip install pandas scikit-learn nltk streamlit matplotlib
```

### 6. Train the Model

```bash
python train_model.py
```

### 7. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## Applications

* Disaster-related tweet classification.
* Social media data analysis.
* Emergency information monitoring.
* Natural Language Processing research.
* Machine learning education and experimentation.

## Future Enhancements

* Real-time social media data integration.
* Classification of newly submitted tweets.
* Improved classification accuracy.
* Geographical visualization of disaster-related tweets.
* Automated alerts for potentially critical tweets.
* Integration of advanced NLP and deep learning models.

## Limitations

* The current system uses a predefined dataset.
* The dashboard does not collect live tweets.
* Classification predictions may contain errors.
* The model may not correctly identify every disaster-related tweet.
* The system is intended for educational purposes and should not be used as the sole source for emergency decisions.

## Disclaimer

This project is developed for academic and educational purposes. The classification results are generated by a machine learning model and may not always be accurate. The system is not a replacement for official disaster management or emergency response services.

## Project Information

Project Title: AI-Based Disaster Tweet Classification and Monitoring Dashboard

Domain: Artificial Intelligence and Machine Learning

Programming Language: Python

Framework: Streamlit

Repository: https://github.com/Elakyaa-07/Disaster-Tweet-Classification
