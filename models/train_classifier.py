import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cloudpickle

# Download NLTK resources and upgrade if necessary
nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)
nltk.download('punkt')  # Ensure punkt is properly downloaded

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def load_data(database_filepath):
    """Load and prepare data from SQLite database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, Y, Y.columns.tolist()


def tokenize(text):
    """Advanced text processing pipeline"""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Avoid sentence splitting
    tokens = word_tokenize(text, preserve_line=True)
    return [lemmatizer.lemmatize(w.strip()) for w in tokens if w not in stop_words]


def build_model():
    """Construct optimized pipeline with validated best parameters"""
    return Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize,
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=50,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
        ))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    """Comprehensive model evaluation."""
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'\nCategory: {category}')
        print(classification_report(
            Y_test[category],
            Y_pred[:, i]
        ))


def save_model(model, model_filepath):
    """Persist trained model using robust serialization"""
    with open(model_filepath, 'wb') as f:
        cloudpickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data from {database_filepath}...')
        X, Y, categories = load_data(database_filepath)

        print('Splitting data...')
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        print('Building optimized model...')
        model = build_model()

        print('Training final model...')
        model.fit(X_train, Y_train)

        print('Evaluating performance...')
        evaluate_model(model, X_test, Y_test, categories)

        print(f'Saving model to {model_filepath}...')
        save_model(model, model_filepath)

        print('Training completed successfully!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
