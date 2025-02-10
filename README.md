# Disaster Response Pipeline Project

## Overview
This project builds a web application to classify disaster messages into relevant categories for emergency response efforts. A machine learning model is trained on a dataset of disaster messages to automate classification and aid in disaster relief operations.

## Setup Instructions

### 1. Install Dependencies
Ensure the following Python packages are installed:
```sh
pip install pandas numpy sqlalchemy nltk scikit-learn flask plotly wordcloud joblib
```

### 2. Prepare the Database and Model
Execute the following scripts in the project's root directory:
- **Run ETL pipeline:** Cleans the data and stores it in a SQLite database.
  ```sh
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```
- **Run ML pipeline:** Trains the model and saves it as a pickle file.
  ```sh
  python models/train_classifier.py models/classifier.pkl
  ```

### 3. Launch the Web Application
Start the web app by running:
```sh
python run.py
```
Then, access the app in your browser at:
```
http://0.0.0.0:3001/
```

## Repository Structure
```
project-root/
├── app/
│   ├── run.py                # Main Flask application
│   ├── templates/
│   │   ├── master.html       # Main web page template
│   │   ├── go.html           # Classification result template
├── data/
│   ├── process_data.py       # ETL pipeline script
│   ├── disaster_messages.csv # Raw dataset - messages
│   ├── disaster_categories.csv # Raw dataset - categories
│   ├── DisasterResponse.db   # Processed SQLite database
├── models/
│   ├── train_classifier.py   # Machine learning pipeline script
│   ├── classifier.pkl        # Trained classification model
├── README.md                 # Project documentation
```

## Web Application Features
The web app includes the following data visualizations:
1. **Message Genres Distribution** - A bar chart showing the distribution of messages by genre (e.g., direct, social, news).
2. **Category Counts** - A bar chart displaying the number of messages per category (e.g., related, request, offer).
3. **Top 10 Disaster Categories** - A bar chart highlighting the most common disaster-related message categories.
4. **Most Used Words** - A bar chart and word cloud visualizing the most frequent words in the messages.

## Machine Learning Pipeline
The machine learning pipeline uses `GridSearchCV` for hyperparameter tuning to find the best model parameters. This ensures that the model is optimized for performance.

## Version Control and Documentation
- The project is managed using Git and stored in a GitHub repository.
- The repository includes multiple commits reflecting the development process.
- The code follows PEP8 guidelines with meaningful variable/function names and proper documentation via comments and docstrings.

## Contribution
If you'd like to contribute, feel free to submit pull requests or raise issues on GitHub!

You can find the project on GitHub at: [GitHub Repository](https://github.com/Bryne-ra/legendary-spoon)

---
Developed as part of the Disaster Response Pipeline Project.



