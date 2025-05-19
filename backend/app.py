import os
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <- forces a non-GUI backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, redirect, url_for, render_template
from sklearn.linear_model import LinearRegression
import numpy as np
import openai
import sys
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="../frontend")
CORS(app, resources={r"/*": {"origins": "https://hariprashad-ravikumar.github.io"}})
@app.route("/")
def home():
    return render_template("index.html", summary="", log="", forecast="", plot_url=None)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
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
    log_print("⚙️ Converting X (dates) to numeric format...")
    df['X'] = pd.to_datetime(df['X'], errors='coerce')
    df.dropna(inplace=True)
    X = df['X'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df['Y'].values

    if len(X) == 0:
        log_print("❌ No valid data to fit the model.")
        return render_template("index.html", summary="No valid data found.", log=log_stream.getvalue(), forecast="N/A", plot_url=None)

    model = LinearRegression()
    model.fit(X, y)
    log_print("🤖 Model trained.")

    # OpenAI Summary
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize this dataset:"},
            {"role": "user", "content": df.head(10).to_csv()}
        ]
     )

    summary = response.choices[0].message.content

    log_print("🧠 OpenAI Summary generated.")
    return jsonify({"summary": summary, "log": log_stream.getvalue(), "forecast": "Submit future x-values below to get predictions.", "plot_url": "/static/plot.png"})


@app.route("/predict", methods=["POST"])
def predict():
    future_x = request.form.get("future_x")

    try:
        # Convert input strings to ordinal integers
        values = [
            datetime.strptime(x.strip(), "%Y-%m-%d").toordinal()
            for x in future_x.split(",")
        ]
    except ValueError:
        log_print("❌ Invalid date format. Use YYYY-MM-DD.")
        return render_template(
            "index.html",
            summary="",
            log=log_stream.getvalue(),
            forecast="Invalid date format. Use YYYY-MM-DD.",
            plot_url=None
        )

    # Reload most recent uploaded file
    filename = os.listdir(UPLOAD_FOLDER)[0]
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename))
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

    return jsonify({ "forecast": result, "log": log_stream.getvalue(), "plot_url": "/static/plot.png" })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)

