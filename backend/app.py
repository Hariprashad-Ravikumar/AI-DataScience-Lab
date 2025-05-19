import os
import io
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np
import openai
from datetime import datetime
import tempfile
import atexit

# === CONFIG ===
BACKEND_BASE_URL = "https://ai-dslab-backend-cpf2feachnetbbck.westus-01.azurewebsites.net"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use a secure temporary directory
TEMP_DIR = tempfile.TemporaryDirectory()
UPLOAD_FOLDER = TEMP_DIR.name
PLOT_PATH = os.path.join(UPLOAD_FOLDER, "plot.png")

# Cleanup temp directory on shutdown
@atexit.register
def cleanup_temp_dir():
    TEMP_DIR.cleanup()

# Capture logs
log_stream = io.StringIO()

def log_print(*args):
    print(*args, file=log_stream)
    sys.stdout.flush()

# Root route for health check
@app.route("/")
def index():
    return jsonify({"message": "✅ AI DataScience Backend is running on Azure."})

# Handle file upload and processing
@app.route("/upload", methods=["POST"])
def upload_file():
    log_stream.truncate(0)
    log_stream.seek(0)

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    log_print("📦 File uploaded:", file.filename)

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
    plt.savefig(PLOT_PATH)
    plt.close()
    log_print("📊 Scatter plot saved.")

    # Fit model
    df['X'] = pd.to_datetime(df['X'], errors='coerce')
    df.dropna(inplace=True)
    X = df['X'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df['Y'].values

    if len(X) == 0:
        log_print("❌ No valid data to fit the model.")
        return jsonify({
            "summary": "No valid data found.",
            "log": log_stream.getvalue(),
            "forecast": "N/A",
            "plot_url": None
        })

    model = LinearRegression()
    model.fit(X, y)
    log_print("🤖 Model trained.")

    # OpenAI summary
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this dataset:"},
                {"role": "user", "content": df.head(10).to_csv()}
            ]
        )
        summary = response.choices[0].message.content
        log_print("🧠 OpenAI Summary generated.")
    except Exception as e:
        summary = "OpenAI summarization failed."
        log_print("❌ OpenAI error:", str(e))

    return jsonify({
        "summary": summary,
        "log": log_stream.getvalue(),
        "forecast": "Submit future x-values below to get predictions.",
        "plot_url": f"{BACKEND_BASE_URL}/plot.png"
    })

# Serve the generated plot
@app.route("/plot.png")
def serve_plot():
    return send_file(PLOT_PATH, mimetype="image/png")

# Handle prediction requests
@app.route("/predict", methods=["POST"])
def predict():
    future_x = request.form.get("future_x")
    if not future_x:
        return jsonify({
            "forecast": "No future values provided.",
            "log": log_stream.getvalue(),
            "plot_url": None
        })

    try:
        values = [datetime.strptime(x.strip(), "%Y-%m-%d").toordinal()
                  for x in future_x.split(",")]
    except ValueError:
        log_print("❌ Invalid date format. Use YYYY-MM-DD.")
        return jsonify({
            "forecast": "Invalid date format. Use YYYY-MM-DD.",
            "log": log_stream.getvalue(),
            "plot_url": None
        })

    try:
        files = os.listdir(UPLOAD_FOLDER)
        if not files:
            raise FileNotFoundError("No uploaded file found.")
        latest_file = max(
            [os.path.join(UPLOAD_FOLDER, f) for f in files if f.endswith(".csv")],
            key=os.path.getctime
        )
        df = pd.read_csv(latest_file)
        df.dropna(inplace=True)
        df.columns = ['X', 'Y']
        df['X'] = pd.to_datetime(df['X'], errors='coerce')
        df.dropna(inplace=True)
        X = df['X'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = df['Y'].values

        model = LinearRegression()
        model.fit(X, y)

        predicted = model.predict(np.array(values).reshape(-1, 1))
        result = {
            datetime.fromordinal(v).strftime("%Y-%m-%d"): round(p, 2)
            for v, p in zip(values, predicted)
        }

        log_print("🔮 Forecast complete.")
        return jsonify({
            "forecast": result,
            "log": log_stream.getvalue(),
            "plot_url": f"{BACKEND_BASE_URL}/plot.png"
        })

    except Exception as e:
        log_print("❌ Prediction failed:", str(e))
        return jsonify({
            "forecast": "Prediction failed.",
            "log": log_stream.getvalue(),
            "plot_url": None
        })
