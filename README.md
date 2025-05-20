# AI-DataScience-Lab

**AI-DataScience-Lab** is an end-to-end forecasting web application designed to upload CSV datasets, clean and analyze them using Python libraries, generate visualizations and predictive models with `scikit-learn`, and summarize the dataset using OpenAI’s GPT-3.5 API.

The frontend is hosted on **GitHub Pages**, and the backend is deployed on **Azure App Service**, creating a scalable and professional architecture suitable for real-world AI and data science workflows.

---

## 🌐 Live Demo

- **Frontend (GitHub Pages):** [https://hariprashad-ravikumar.github.io/AI-DataScience-Lab/](https://hariprashad-ravikumar.github.io/AI-DataScience-Lab/)
- **Backend (Azure)**

---

## ⚙️ Features

- Upload CSV files with two columns: `X` (dates) and `Y` (numerical values)
- Cleans data using `pandas`, removes invalid entries
- Generates a scatter plot using `matplotlib`
- Converts date strings to ordinal format and trains a `LinearRegression` model with `scikit-learn`
- Uses **OpenAI API** (GPT-3.5-turbo) to summarize the uploaded dataset
- Predicts future `Y` values for user-supplied future `X` (date) values
- Secure HTTPS communication across GitHub and Azure (CORS-enabled)
- Temporary file storage using Python's `tempfile`, cleaned automatically on restart

---

## 📊 Technical Workflow

### 1. **Frontend (GitHub Pages)**

- HTML + JavaScript app with forms to:
  - Upload CSV data
  - Request future predictions
- Communicates with the backend via `fetch()` using HTTPS POST requests
- Displays:
  - Processing log
  - OpenAI-generated summary
  - Forecast output
  - Auto-generated plot image

### 2. **Backend (Azure App Service - Python Flask)**

- **Routes:**
  - `POST /upload`: Handles file uploads, data cleaning, modeling, summary generation
  - `POST /predict`: Accepts future dates, returns predictions
  - `GET /plot.png`: Serves saved scatter plot image

### 3. **Processing Pipeline**

- **Step 1: Data Cleaning**
  - Reads CSV using `pandas`
  - Drops NA values and converts `X` to datetime format

- **Step 2: Visualization**
  - Uses `matplotlib` to generate scatter plot
  - Plot saved to a temporary directory and served on request

- **Step 3: Modeling**
  - Uses `scikit-learn` `LinearRegression` to fit `X` (date ordinal) → `Y`
  - Model used to predict future values based on user input

- **Step 4: Summarization**
  - Sends cleaned dataset (via `.head(10).to_csv()`) to OpenAI GPT-3.5 API
  - Summary generated and returned to frontend

---

## 🛠️ Tech Stack

| Layer     | Technology                               |
|-----------|-------------------------------------------|
| Frontend  | HTML, JavaScript, GitHub Pages            |
| Backend   | Flask, Azure App Service                  |
| ML Tools  | `pandas`, `scikit-learn`, `matplotlib`    |
| AI        | OpenAI GPT-3.5 (`openai` Python SDK)      |
| Storage   | Python `tempfile` for secure cleanup      |
| Deployment| Gunicorn + Azure Linux App Container      |

---

## 🔐 Security and Performance

- Uses `flask-cors` to securely allow cross-origin requests from GitHub Pages
- All requests are served over HTTPS
- Files and plots are saved temporarily and deleted automatically on app shutdown using `tempfile.TemporaryDirectory` and `atexit`

---

## 🚀 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Hariprashad-Ravikumar/AI-DataScience-Lab.git
   cd AI-DataScience-Lab/backend
