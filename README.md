# Explainable Fake News Detection System

This repository contains a complete Python-based system for detecting and explaining fake news classifications. The system uses machine learning to analyze news content and provides explanations for its predictions.

## Features

- Text preprocessing and feature extraction
- ML-based classification (Random Forest and Logistic Regression options)
- Explainable AI using LIME (Local Interpretable Model-agnostic Explanations)
- User-friendly web interface built with Streamlit
- Support for both training new models and using pre-trained models

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install pandas numpy scikit-learn nltk lime shap streamlit matplotlib seaborn
```

3. Download the NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Running the Web Interface

To start the Streamlit web interface:

```bash
streamlit run fake_news_detection_system.py
```

This will open a web browser with the application interface where you can input news content for analysis.

### Using the Datasets

Place the datasets in the same directory as the application:
- merged_dataset.csv
- structured_fake_news.csv

The system will first try to use the merged dataset and fall back to the structured dataset if needed.

### Using the API Programmatically

You can also use the detector directly in your Python code:

```python
from fake_news_detection_system import FakeNewsDetector

# Initialize the detector with your preferred model type
detector = FakeNewsDetector(model_type="random_forest")  # or "logistic_regression"

# Train on your dataset
detector.train("your_dataset.csv")

# Make predictions
news_text = "Your news text here..."
label, confidence = detector.predict(news_text)
print(f"Classification: {label} (Confidence: {confidence:.2%})")

# Get explanations
explanation_text, _ = detector.explain_prediction(news_text)
print(f"Explanation: {explanation_text}")

# Save the trained model
detector.save_model("your_model.pkl")

# Load a pre-trained model
detector.load_model("your_model.pkl")
```

## System Architecture

The system consists of several components:

1. **Preprocessing Module**: Cleans and normalizes text input
2. **ML Model**: Classifies news as real or fake
3. **Explainability Module**: Generates human-understandable explanations
4. **Web Interface**: Provides user-friendly access to the system

## Customization

### Adding New Models

You can extend the `FakeNewsDetector` class to support additional model types:

```python
# In the __init__ method
if self.model_type == "your_new_model":
    model = YourNewModelClass()
```

### Improving Explanations

The explanation system uses LIME by default. You can customize the explanations by modifying the `explain_prediction` method in the `FakeNewsDetector` class.

## Limitations

- The model's accuracy is dependent on the quality and diversity of the training data
- Very short or ambiguous texts may lead to less reliable classifications
- Models can exhibit bias present in the training data
- Always use this tool as an aid rather than the sole determinant of news credibility

## Contributing

Contributions to improve the system are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is released under the MIT License.