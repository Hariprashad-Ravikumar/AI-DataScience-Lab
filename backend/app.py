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
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import openai
from datetime import datetime
import tempfile
import atexit
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# === CONFIG ===
BACKEND_BASE_URL = "https://ai-dslab-backend-cpf2feachnetbbck.westus-01.azurewebsites.net"

app = Flask(__name__)
CORS(app)

TEMP_DIR = tempfile.TemporaryDirectory()
UPLOAD_FOLDER = TEMP_DIR.name
PLOT_PATH = os.path.join(UPLOAD_FOLDER, "plot.png")
FORECAST_PLOT_PATH = os.path.join(UPLOAD_FOLDER, "forecast_plot.png")
REPORT_PATH = os.path.join(UPLOAD_FOLDER, "summary_report.pdf")
CSV_CACHE = os.path.join(UPLOAD_FOLDER, "cached_upload.csv")
cached_summary = ""

@atexit.register
def cleanup_temp_dir():
    TEMP_DIR.cleanup()

log_stream = io.StringIO()

def log_print(*args):
    print(*args, file=log_stream)
    sys.stdout.flush()

# === PDF REPORT GENERATOR ===
def generate_pdf_report(summary, r2, mse, forecast_dict):
    c = canvas.Canvas(REPORT_PATH, pagesize=letter)
    width, height = letter
    text = c.beginText(1 * inch, height - 1 * inch)

    text.setFont("Helvetica-Bold", 14)
    text.textLine("AI-DataScience-Lab Report")
    text.textLine("Created by Hariprashad Ravikumar")
    text.textLine("https://github.com/Hariprashad-Ravikumar")
    text.textLine("")
    text.setFont("Helvetica", 10)
    text.textLine(f"Model: Linear Regression")
    text.textLine(f"R² Score: {r2:.4f}")
    text.textLine(f"MSE: {mse:.4f}")
    text.textLine("")

    text.setFont("Helvetica-Bold", 12)
    text.textLine("OpenAI Data Summary:")
    text.setFont("Helvetica", 9)
    for line in summary.splitlines():
        for wrapped in textwrap.wrap(line, width=100):
            text.textLine(wrapped)

    text.textLine("")
    text.setFont("Helvetica-Bold", 12)
    text.textLine("Forecast Results:")
    text.setFont("Helvetica", 10)
    for k, v in forecast_dict.items():
        text.textLine(f"{k}: {v}")

    c.drawText(text)
    if os.path.exists(FORECAST_PLOT_PATH):
        c.drawImage(FORECAST_PLOT_PATH, 1 * inch, 1 * inch, width=5.5 * inch, preserveAspectRatio=True)
    c.save()

# === PLOT FORECAST (READABLE X) ===
def plot_forecast_with_axis(X, y, model, values_parsed, y_future, use_dates):
    x_min, x_max = min(X.min(), values_parsed.min()), max(X.max(), values_parsed.max())
    x_plot = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    y_plot = model.predict(x_plot)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label='Training Data', alpha=0.6)
    plt.plot(x_plot, y_plot, color='blue', label='Linear Regression')
    plt.scatter(values_parsed, y_future, color='red', label='Forecast', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Forecast with Linear Regression')

    if use_dates:
        ticks = np.linspace(x_min, x_max, 6, dtype=int)
        labels = [datetime.fromordinal(t).strftime('%Y-%m-%d') for t in ticks]
        plt.xticks(ticks, labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PATH)
    plt.close()

# === ROUTES ===
@app.route("/get-columns", methods=["POST"])
def get_columns():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400
    df = pd.read_csv(file)
    df.to_csv(CSV_CACHE, index=False)
    return jsonify({"columns": df.columns.tolist()})

@app.route("/upload", methods=["POST"])
def upload_file():
    global cached_summary
    log_stream.truncate(0)
    log_stream.seek(0)

    file = request.files.get("file")
    x_col = request.form.get("x_column")
    y_col = request.form.get("y_column")

    if file is None or not x_col or not y_col:
        return jsonify({"error": "Missing file or column selection."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)
    df.to_csv(CSV_CACHE, index=False)

    if x_col not in df.columns or y_col not in df.columns:
        return jsonify({"error": f"'{x_col}' or '{y_col}' not in dataset."}), 400

    df = df[[x_col, y_col]].dropna()
    df.columns = ['X', 'Y']
    log_print("Data Cleaned:\n\n", df.head())

    # Save scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(df['X'], df['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    df['X_date'] = pd.to_datetime(df['X'], errors='coerce')
    use_dates = df['X_date'].notna().sum() >= len(df) // 2
    if use_dates:
        df = df.dropna(subset=['X_date'])
        X = df['X_date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    else:
        X = df['X'].astype(float).values.reshape(-1, 1)
    y = df['Y'].astype(float).values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    log_print(f"\nModel Trained:\nR² = {r2:.4f}, MSE = {mse:.4f}")

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
        cached_summary = summary
    except Exception as e:
        summary = "OpenAI summarization failed."
        cached_summary = summary
        log_print("OpenAI error:", str(e))

    return jsonify({
        "summary": summary,
        "log": log_stream.getvalue(),
        "forecast": "Submit future values below to get predictions.",
        "r2_score": round(r2, 4),
        "mse": round(mse, 4),
        "plot_url": f"{BACKEND_BASE_URL}/plot.png"
    })

@app.route("/predict", methods=["POST"])
def predict():
    future_x = request.form.get("future_x")
    x_col = request.form.get("x_column")
    y_col = request.form.get("y_column")

    if not future_x or not x_col or not y_col:
        return jsonify({"forecast": "Missing input values or column selection."}), 400

    try:
        values = future_x.split(",")
        numeric_vals, date_vals = [], []
        for x in values:
            try:
                date_vals.append(datetime.strptime(x.strip(), "%Y-%m-%d").toordinal())
            except:
                numeric_vals.append(float(x.strip()))
        values_parsed = np.array(date_vals if date_vals else numeric_vals).reshape(-1, 1)
    except Exception as e:
        return jsonify({"forecast": f"Invalid input: {str(e)}"}), 400

    df = pd.read_csv(CSV_CACHE)
    if x_col not in df.columns or y_col not in df.columns:
        return jsonify({"forecast": "Selected columns not found."}), 400

    df = df[[x_col, y_col]].dropna()
    df.columns = ['X', 'Y']
    df['X_date'] = pd.to_datetime(df['X'], errors='coerce')
    use_dates = df['X_date'].notna().sum() >= len(df) // 2
    if use_dates:
        df = df.dropna(subset=['X_date'])
        X = df['X_date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    else:
        X = df['X'].astype(float).values.reshape(-1, 1)
    y = df['Y'].astype(float).values

    model = LinearRegression()
    model.fit(X, y)
    y_future = model.predict(values_parsed)
    result = {
        datetime.fromordinal(int(x)).strftime("%Y-%m-%d") if use_dates else float(x): round(p, 2)
        for x, p in zip(values_parsed.flatten(), y_future)
    }

    plot_forecast_with_axis(X, y, model, values_parsed, y_future, use_dates)
    generate_pdf_report(cached_summary, r2_score(y, model.predict(X)), mean_squared_error(y, model.predict(X)), result)

    return jsonify({
        "forecast": result,
        "log": log_stream.getvalue(),
        "plot_url": f"{BACKEND_BASE_URL}/plot.png",
        "forecast_plot_url": f"{BACKEND_BASE_URL}/forecast_plot.png"
    })

@app.route("/plot.png")
def serve_plot():
    return send_file(PLOT_PATH, mimetype="image/png")

@app.route("/forecast_plot.png")
def serve_forecast_plot():
    return send_file(FORECAST_PLOT_PATH, mimetype="image/png")

@app.route("/download-report", methods=["GET"])
def download_report():
    return send_file(REPORT_PATH, as_attachment=True, download_name="report.pdf")
