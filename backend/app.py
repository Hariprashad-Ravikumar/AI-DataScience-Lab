import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, redirect, url_for
from sklearn.linear_model import LinearRegression
import numpy as np
import openai
import sys

app = Flask(__name__)
UPLOAD_FOLDER = "backend/uploads"
STATIC_FOLDER = "backend/static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

log_stream = io.StringIO()

def log_print(*args):
    print(*args, file=log_stream)
    sys.stdout.flush()

@app.route("/upload", methods=["POST"])
def upload_file():
    log_stream.truncate(0)
    log_stream.seek(0)

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    log_print("📦 File uploaded:", file.filename)

    # Load and clean
    log_print("🔧 Importing pandas...")
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.columns = ['X', 'Y']
    log_print("🔍 Cleaned Data:", df.head())

    # Plot
    plt.figure()
    plt.scatter(df['X'], df['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')
    plt.savefig(f"{STATIC_FOLDER}/plot.png")
    plt.close()
    log_print("📊 Scatter plot saved.")

    # Model
    log_print("⚙️ Importing scikit-learn...")
    X = pd.to_numeric(df['X'], errors='coerce').dropna().values.reshape(-1, 1)
    y = df['Y'].values
    model = LinearRegression()
    model.fit(X, y)
    log_print("🤖 Model trained.")

    # OpenAI Summary
    openai.api_key = os.getenv("OPENAI_API_KEY")
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Summarize this dataset:"},
                  {"role": "user", "content": df.head(10).to_csv()}]
    ).choices[0].message.content

    log_print("🧠 OpenAI Summary generated.")

    return {
        "summary": summary,
        "log": log_stream.getvalue(),
        "forecast": "Submit future x-values below to get predictions."
    }

@app.route("/predict", methods=["POST"])
def predict():
    future_x = request.form.get("future_x")
    values = [float(x.strip()) for x in future_x.split(",")]
    model = LinearRegression()

    # Dummy fit for now
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0]))
    df.dropna(inplace=True)
    df.columns = ['X', 'Y']
    X = pd.to_numeric(df['X'], errors='coerce').dropna().values.reshape(-1, 1)
    y = df['Y'].values
    model.fit(X, y)

    prediction = model.predict(np.array(values).reshape(-1, 1))
    return {
        "forecast": dict(zip(values, prediction.tolist()))
    }

if __name__ == "__main__":
    app.run(debug=True)
