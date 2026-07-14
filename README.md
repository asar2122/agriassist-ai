# 🌱 AgriAssist AI

AgriAssist AI is an offline smart agriculture assistant that combines Retrieval-Augmented Generation (RAG), machine learning, deep learning, and a Streamlit web interface to provide agriculture-related assistance.

The system includes an agriculture chatbot, crop recommendation, fertilizer recommendation, rice leaf disease detection, market price estimation, and chat history management.

The project is designed to run locally and uses Ollama for local large language model inference.

---

## 📌 Project Overview

Farmers and agriculture users often need access to information about crop cultivation, fertilizers, plant diseases, suitable crops, and market prices.

AgriAssist AI combines multiple AI and machine learning technologies into a single application.

The system provides the following features:

- Agriculture chatbot using RAG
- Local agriculture knowledge base
- Crop recommendation
- Fertilizer recommendation
- Rice leaf disease detection
- Market price estimation
- Chat history storage
- Automated testing

---

## ✨ Features

### 🤖 Agriculture Chatbot

The agriculture chatbot answers questions using information retrieved from a local agriculture knowledge base.

The chatbot uses:

- Sentence Transformers for embeddings
- FAISS for vector similarity search
- Ollama for local LLM inference
- Retrieval-Augmented Generation (RAG)
- Intent classification
- Source attribution

Example questions:

```text
What are the symptoms of rice blast?

How can rice blast be managed?

How is rice cultivated?
```

For questions outside the available agriculture knowledge base, the chatbot returns an appropriate fallback response.

---

### 🌾 Crop Recommendation

The crop recommendation module recommends a suitable crop based on soil nutrients and environmental conditions.

Input features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The model uses a Random Forest Classifier.

The crop dataset contains:

- 2,200 records
- 22 crop classes

The trained model achieved approximately:

```text
Accuracy: 99.55%
```

Supported crop classes include:

- Rice
- Maize
- Chickpea
- Kidney Beans
- Pigeon Peas
- Moth Beans
- Mung Bean
- Black Gram
- Lentil
- Pomegranate
- Banana
- Mango
- Grapes
- Watermelon
- Muskmelon
- Apple
- Orange
- Papaya
- Coconut
- Cotton
- Jute
- Coffee

---

### 💊 Fertilizer Recommendation

The fertilizer recommendation module recommends a suitable fertilizer based on soil, crop, and environmental information.

Input features:

- Temperature
- Humidity
- Soil Moisture
- Nitrogen
- Potassium
- Phosphorus
- Soil Type
- Crop Type

The module uses a Random Forest Classifier.

The fertilizer dataset contains:

```text
99 records
```

The trained model achieved approximately:

```text
Accuracy: 95%
```

Example fertilizer classes include:

- Urea
- DAP
- 14-35-14
- 28-28
- 17-17-17
- 20-20
- 10-26-26

---

### 🦠 Rice Disease Detection

The disease detection module analyzes uploaded rice leaf images and predicts possible rice diseases.

The module uses transfer learning with a ResNet18 convolutional neural network.

The disease dataset contains:

```text
3,829 images
6 classes
```

Supported classes:

- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf Scald
- Sheath Blight

The model achieved approximately:

```text
Best Validation Accuracy: 85.49%
```

The Streamlit interface displays:

- Predicted disease
- Prediction confidence
- Model information
- Number of supported classes

---

### 💰 Market Price Estimation

The market price module estimates the modal price of an agricultural commodity.

Input features:

- Month
- Commodity
- State
- District
- Average Minimum Price
- Average Maximum Price

The module uses a Random Forest Regressor.

The market dataset contains:

```text
2,810 records
```

Model evaluation results:

```text
Mean Absolute Error: 61.5973
Root Mean Squared Error: 283.3085
R² Score: 0.9972
```

> Note: This module estimates modal market prices using minimum and maximum price information. It should be considered a market price estimation system rather than a true future price forecasting system.

---

### 📚 Chat History

Chat conversations are stored locally using SQLite.

The Chat History page displays:

- User questions
- AgriAssist responses
- Date and time

---

## 🧠 System Architecture

```text
                         AgriAssist AI
                               |
                         Streamlit UI
                               |
        -------------------------------------------------
        |              |              |                 |
    Chatbot           Crop        Fertilizer         Disease
      RAG         Recommendation Recommendation      Detection
        |              |              |                 |
      FAISS       Random Forest  Random Forest        ResNet18
        |
Sentence Transformers
        |
Local Knowledge Base
        |
      Ollama

                               |
                        Market Estimation
                               |
                    Random Forest Regressor

                               |
                            SQLite
                               |
                         Chat History
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Web Framework

- Streamlit

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- Random Forest Regressor
- Joblib

### Deep Learning

- PyTorch
- Torchvision
- ResNet18
- Transfer Learning

### Retrieval-Augmented Generation

- Sentence Transformers
- FAISS
- Ollama

### Data Processing

- Pandas
- NumPy

### Image Processing

- Pillow
- Torchvision Transforms

### Document Processing

- PyMuPDF

### Database

- SQLite

### Testing

- Pytest

---

## 📁 Project Structure

```text
agriassist-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── chatbot/
│   ├── __init__.py
│   ├── chatbot.py
│   └── intent_classifier.py
│
├── crop_recommendation/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
├── fertilizer_recommendation/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
├── disease_detection/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── image_processing.py
│
├── market_prediction/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
├── database/
│   ├── __init__.py
│   └── database.py
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   └── ...
│
├── scripts/
│   └── build_vector_database.py
│
├── data/
│   ├── crop_data.csv
│   ├── fertilizer_data.csv
│   ├── market_data.csv
│   ├── disease_dataset/
│   └── knowledge_base/
│
├── models/
│   ├── crop_model.pkl
│   ├── fertilizer_model.pkl
│   ├── disease_model.pth
│   └── market_model.pkl
│
├── vector_database/
│   └── faiss_index/
│       ├── agriassist.index
│       └── metadata.json
│
└── tests/
    ├── __init__.py
    ├── test_chatbot.py
    ├── test_crop_recommendation.py
    ├── test_fertilizer_recommendation.py
    ├── test_disease_detection.py
    └── test_market_prediction.py
```

> The exact files inside the `rag/` directory may vary depending on the implementation.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd agriassist-ai
```

---

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
streamlit
pandas
numpy
scikit-learn
pytest
PyMuPDF
sentence-transformers
faiss-cpu
torch
torchvision
Pillow
joblib
requests
```

---

## 🦙 Ollama Setup

The agriculture chatbot uses Ollama for local LLM inference.

Install Ollama and download the model configured in the project.

For example:

```bash
ollama pull llama3.2:3b
```

Verify the available models:

```bash
ollama list
```

Test the model:

```bash
ollama run llama3.2:3b "Hello"
```

Make sure Ollama is available before running chatbot integration tests.

---

## 📚 Building the FAISS Vector Database

Agriculture documents are stored inside:

```text
data/knowledge_base/
```

Example:

```text
data/
└── knowledge_base/
    ├── crop_guides/
    │   └── rice_cultivation.txt
    │
    └── disease_guides/
        └── rice_diseases.txt
```

Build the vector database with:

```bash
python -m scripts.build_vector_database
```

This generates the FAISS index and metadata used by the agriculture chatbot.

---

## 🤖 Training the Models

### Train the Crop Recommendation Model

```bash
python -m crop_recommendation.train
```

The trained model is saved as:

```text
models/crop_model.pkl
```

Test the model:

```bash
python -m crop_recommendation.predict
```

---

### Train the Fertilizer Recommendation Model

```bash
python -m fertilizer_recommendation.train
```

The trained model is saved as:

```text
models/fertilizer_model.pkl
```

Test the model:

```bash
python -m fertilizer_recommendation.predict
```

---

### Train the Disease Detection Model

```bash
python -m disease_detection.train
```

The trained model is saved as:

```text
models/disease_model.pth
```

Training time depends on the available hardware.

The model can run on CPU, but GPU acceleration can significantly reduce training time.

---

### Train the Market Price Estimation Model

```bash
python -m market_prediction.train
```

The trained model is saved as:

```text
models/market_model.pkl
```

Test the model:

```bash
python -m market_prediction.predict
```

---

## ▶️ Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

The application will start locally.

Open the local address displayed by Streamlit in a web browser.

Example:

```text
http://localhost:8501
```

---

## 🧪 Running Tests

Run the complete automated test suite:

```bash
python -m pytest tests -v
```

Current test results:

```text
11 passed
0 failed
```

The tests cover:

- Intent classification
- Relevant agriculture chatbot questions
- Out-of-domain chatbot questions
- Empty chatbot questions
- Crop recommendation
- Fertilizer recommendation
- Disease detection
- Market price estimation

---

## 📊 Model Performance

| Module | Algorithm | Dataset Size | Performance |
|---|---|---:|---:|
| Crop Recommendation | Random Forest Classifier | 2,200 rows | 99.55% accuracy |
| Fertilizer Recommendation | Random Forest Classifier | 99 rows | 95% accuracy |
| Disease Detection | ResNet18 | 3,829 images | 85.49% validation accuracy |
| Market Price Estimation | Random Forest Regressor | 2,810 rows | R² = 0.9972 |

---

## ⚠️ Limitations

The current project has several limitations:

- The agriculture chatbot can answer only from the available local knowledge base.
- The knowledge base currently contains a limited number of agriculture documents.
- Crop recommendations depend on the quality and coverage of the training dataset.
- The fertilizer recommendation dataset is relatively small.
- Disease detection currently supports six rice leaf classes.
- The disease model may classify unrelated images as one of the known classes.
- Disease predictions should not be treated as professional agricultural diagnoses.
- Market price estimation is not true future market forecasting.
- Model performance may differ on unseen real-world data.
- The application depends on a locally available Ollama model for chatbot responses.

---

## 🚀 Future Improvements

Possible future improvements include:

- Expand the agriculture knowledge base.
- Add additional crops and agriculture documents.
- Improve RAG retrieval and reranking.
- Add multilingual chatbot support.
- Add speech input and voice responses.
- Add weather information.
- Add location-aware agriculture recommendations.
- Expand the disease dataset.
- Support disease detection for additional crops.
- Add unknown-image rejection to the disease detector.
- Improve fertilizer recommendation using larger datasets.
- Develop genuine time-series market price forecasting.
- Add model explainability.
- Improve the Streamlit user interface.
- Add authentication and user accounts.
- Add Docker support.
- Add continuous integration testing.
- Deploy the application.

---

## 🔒 Offline AI Capabilities

AgriAssist AI is designed around local AI components.

The project can use:

- Local FAISS vector search
- Local Sentence Transformer embeddings
- Local machine learning models
- Local PyTorch disease detection
- Local SQLite storage
- Local Ollama LLM inference

This architecture can reduce dependence on external cloud AI APIs.

---

## 📌 Disclaimer

AgriAssist AI is an educational and experimental project.

Crop, fertilizer, disease, and market outputs are generated by machine learning and AI models and may not always be accurate.

Users should consult qualified agricultural professionals and locally approved agricultural guidance before making important farming, pesticide, fertilizer, disease-management, or financial decisions.

---

## 👨‍💻 Author

Developed as an AI and Machine Learning agriculture assistance project.

---

## 📄 License

This project is intended for educational and research purposes.

Add a specific open-source license file before distributing or licensing the project publicly.