# BERT Fake News Detection with Groq API Explainability

This project enhances a BERT-based fake news detection system with explainability features powered by the Groq API. After classifying text as real or fake news, the system generates a human-readable explanation of the classification.

## Features

- **BERT-based Classification**: Detect fake news with high accuracy using a fine-tuned BERT model
- **Groq API Integration**: Generate explanations for why text was classified as real or fake news
- **Feature Analysis**: Extracts and analyzes statistical patterns in text that may indicate fake news
- **Streamlit Web Interface**: User-friendly interface for text analysis and visualization
- **Command-line Testing Tool**: Test the integration with sample or custom texts

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Streamlit (for web interface)
- A Groq API key (sign up at [console.groq.com](https://console.groq.com/signup))

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install torch transformers streamlit nltk scikit-learn matplotlib seaborn pandas requests python-dotenv
   ```
3. Download the NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Set up Groq API

1. Sign up for an account at [console.groq.com](https://console.groq.com/signup)
2. Generate an API key from your Groq dashboard
3. Set up your API key in one of these ways:
   - Create a `.env` file with `GROQ_API_KEY=your_api_key_here`
   - Set an environment variable: `export GROQ_API_KEY=your_api_key_here`
   - Pass the API key directly to the scripts using `--api-key` parameter

## Usage

### Web Interface

Run the Streamlit app to analyze news text via a user-friendly interface:

```bash
streamlit run streamlit_groq_integration.py
```

The web interface allows you to:
- Enter news text to analyze
- View classification results with confidence scores
- See AI-generated explanations from Groq (if API key is provided)
- Examine text features that contributed to the classification

### Command Line Testing

Use the test script to quickly analyze texts without the web interface:

```bash
# Test with sample texts
python test_groq_integration.py --model-path bert_fake_news_model

# Test with your own text
python test_groq_integration.py --text "Your news text here" --output results.json

# Specify a different Groq model
python test_groq_integration.py --model llama3-8b-8192 --api-key your_api_key_here
```

## How Groq API Integration Works

The system follows these steps to generate explanations:

1. The BERT model classifies the input text as "Real News" or "Fake News" with a confidence score
2. The system extracts statistical features from the text (if the enhanced model is used)
3. A request is sent to the Groq API with:
   - The original text
   - The classification result
   - The confidence score
   - Feature importance scores (if available)
4. The API returns a detailed explanation of why the text was likely classified this way
5. The explanation is displayed to the user

### API Request Structure

Here's the basic structure of the request sent to Groq's API:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-70b-8192",
    "messages": [
        {
            "role": "system", 
            "content": "You are an expert in media literacy and fake news detection..."
        },
        {
            "role": "user", 
            "content": f"The following news text was classified as '{classification}'..."
        }
    ],
    "temperature": 0.3,
    "max_tokens": 1024
}

response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                       headers=headers, json=payload)
```

## Available Groq Models

You can use different Groq models for explanation generation:

- `llama3-70b-8192` (default, most powerful)
- `llama3-8b-8192` (faster, lighter)
- `mixtral-8x7b-32768` (good for longer contexts)
- `gemma-7b-it`

Specify the model using the `--model` parameter in the test script or by changing the model in the `GroqExplainer` class.

## Best Practices

- Provide a Groq API key for best results
- Use explanations as a guide, not as definitive truth
- Cross-check information with reputable sources
- Consider the context and source of the news
- Be aware that AI models have limitations and may occasionally misclassify content

## Limitations

- The BERT model analyzes text patterns and cannot fact-check specific claims
- Performance depends on the training data and may not catch novel misinformation tactics
- API explanations should be used as an aid to human judgment, not a replacement for critical thinking
- API keys must be kept secure and may incur usage charges from Groq