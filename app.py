import webbrowser
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from fpdf import FPDF
import io
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ─────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin":
            session['username'] = username
            flash("Login Successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session.get('username', 'Guest')
    return render_template('dashboard.html', username=username)

# ─────────────────────────────────────────
# UPLOAD DATASET
# ─────────────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            os.makedirs('datasets', exist_ok=True)
            filepath = os.path.join('datasets', file.filename)
            file.save(filepath)
            session['dataset_path'] = filepath

            df = pd.read_csv(filepath)
            preview = df.head(10).to_html(classes='table-dark', index=False)
            flash("Dataset uploaded successfully!", "success")
            return render_template('preview.html', preview=preview)
        else:
            flash("Please upload a valid CSV file!", "error")
            return redirect(url_for('upload'))
    return render_template('upload.html')

# ─────────────────────────────────────────
# TRAIN MODEL (New route - was missing)
# ─────────────────────────────────────────
@app.route('/testing')
def testing():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    dataset_path = session.get('dataset_path', 'datasets/Metro_Interstate_Traffic_Volume.csv')
    try:
        df = pd.read_csv(dataset_path)
        df = df.dropna()

        le_main = LabelEncoder()
        le_desc = LabelEncoder()
        df['weather_main_enc'] = le_main.fit_transform(df['weather_main'])
        df['weather_description_enc'] = le_desc.fit_transform(df['weather_description'])

        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month

        features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all',
                   'weather_main_enc', 'weather_description_enc',
                   'hour', 'day_of_week', 'month']
        X = df[features]
        y = df['traffic_volume']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test, y_pred), 4)
        r2 = round(r2_score(y_test, y_pred) * 100, 2)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/traffic_model.pkl')
        joblib.dump(le_main, 'models/le_main.pkl')
        joblib.dump(le_desc, 'models/le_desc.pkl')

        session['accuracy'] = r2
        session['mse'] = mse
        session['model_trained'] = 'Random Forest'

        flash("✅ Model trained successfully!", "success")
    except Exception as e:
        flash(f"Training failed: {str(e)}", "error")

    return redirect(url_for('testing_page'))

# ─────────────────────────────────────────
# PREDICTION PAGE
# ─────────────────────────────────────────
@app.route('/testing_page')
def testing_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('testing.html')

# ─────────────────────────────────────────
# MAKE PREDICTION
# ─────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        clouds = float(request.form['clouds'])
        weather_main = request.form['weather_main']
        weather_desc = request.form['weather_desc']
        day = request.form['day']
        date = request.form['date']
        time_val = request.form['time']
        model_name = request.form['model']

        hour = int(time_val.split(':')[0]) if time_val else 0
        days_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        day_of_week = days_map.get(day, 0)
        month = int(date.split('-')[1]) if '-' in date else 1

        le_main = joblib.load('models/le_main.pkl')
        le_desc = joblib.load('models/le_desc.pkl')

        weather_main_enc = 0 if weather_main not in le_main.classes_ else le_main.transform([weather_main])[0]
        weather_desc_enc = 0 if weather_desc not in le_desc.classes_ else le_desc.transform([weather_desc])[0]

        features = [[temp, rain, snow, clouds, weather_main_enc, weather_desc_enc, hour, day_of_week, month]]

        model = joblib.load('models/traffic_model.pkl')
        traffic_volume = round(model.predict(features)[0], 2)

        if traffic_volume < 2000:
            prediction_level = "🟢 Low Traffic"
            prediction_level_pdf = "Low Traffic"
        elif traffic_volume < 4000:
            prediction_level = "🟡 Moderate Traffic"
            prediction_level_pdf = "Moderate Traffic"
        elif traffic_volume < 5500:
            prediction_level = "🟠 High Traffic"
            prediction_level_pdf = "High Traffic"
        else:
            prediction_level = "🔴 Very High Traffic"
            prediction_level_pdf = "Very High Traffic"

        session['last_result'] = {
            'temp': temp, 'rain': rain, 'snow': snow, 'clouds': clouds,
            'weather_main': weather_main, 'weather_desc': weather_desc,
            'day': day, 'date': date, 'time': time_val,
            'model_name': model_name, 'traffic_volume': traffic_volume,
            'prediction_level': prediction_level,
            'prediction_level_pdf': prediction_level_pdf
        }

        return render_template("result.html",
                             temp=temp, rain=rain, snow=snow, clouds=clouds,
                             weather_main=weather_main, weather_desc=weather_desc,
                             day=day, date=date, time=time_val,
                             model_name=model_name, traffic_volume=traffic_volume,
                             prediction_level=prediction_level)

    except FileNotFoundError:
        flash("⚠️ Model not found! Please click 'Click to Train | Test' first.", "error")
        return redirect(url_for('testing_page'))
    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for('testing_page'))

# ─────────────────────────────────────────
# PERFORMANCE ANALYSIS
# ─────────────────────────────────────────
@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    accuracy = session.get('accuracy', 'N/A')
    mse = session.get('mse', 'N/A')
    model_name = session.get('model_trained', 'Not trained yet')
    return render_template('analysis.html', accuracy=accuracy, mse=mse, model_name=model_name)

# ─────────────────────────────────────────
# TRAFFIC CHART
# ─────────────────────────────────────────
@app.route('/chart')
def chart():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        dataset_path = session.get('dataset_path', 'datasets/Metro_Interstate_Traffic_Volume.csv')
        df = pd.read_csv(dataset_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['day'] = df['date_time'].dt.date
        daily = df.groupby('day')['traffic_volume'].mean().reset_index()
        daily = daily.tail(30)
        days = daily['day'].astype(str).tolist()
        daily_avg = [round(v, 2) for v in daily['traffic_volume'].tolist()]
    except:
        days = ["No data"]
        daily_avg = [0]
    return render_template('chart.html', days=days, daily_avg=daily_avg)

# ─────────────────────────────────────────
# LOGOUT
# ─────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────
# DOWNLOAD PDF
# ─────────────────────────────────────────
@app.route('/download_pdf')
def download_pdf():
    if 'username' not in session:
        return redirect(url_for('login'))

    r = session.get('last_result', {})

    # Helper: strip any character outside latin-1 range (e.g. emojis)
    def safe(val):
        return str(val).encode('latin-1', errors='ignore').decode('latin-1')

    # Use emoji-free level if available, fallback to stripping
    level_text = r.get('prediction_level_pdf') or safe(r.get('prediction_level', 'N/A'))

    pdf = FPDF()
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 14, "Traffic Prediction Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Generated by TrafficIQ  |  User: {safe(session.get('username', 'N/A'))}", ln=True, align="C")
    pdf.ln(4)

    # ── Divider ──
    pdf.set_draw_color(30, 80, 160)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Section: Input Parameters ──
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 9, "  Input Parameters", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Arial", size=11)
    rows = [
        ("Temperature (K)",     safe(r.get('temp', 'N/A'))),
        ("Rain (mm/h)",         safe(r.get('rain', 'N/A'))),
        ("Snow (mm/h)",         safe(r.get('snow', 'N/A'))),
        ("Cloud Cover (%)",     safe(r.get('clouds', 'N/A'))),
        ("Weather Condition",   safe(r.get('weather_main', 'N/A'))),
        ("Weather Description", safe(r.get('weather_desc', 'N/A'))),
        ("Day",                 safe(r.get('day', 'N/A'))),
        ("Date",                safe(r.get('date', 'N/A'))),
        ("Time",                safe(r.get('time', 'N/A'))),
        ("Model Used",          safe(r.get('model_name', 'N/A'))),
    ]

    fill = False
    for label, value in rows:
        pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(90, 9, label, border=1, fill=True)
        pdf.cell(100, 9, value, border=1, ln=True, fill=True)
        fill = not fill

    pdf.ln(6)

    # ── Section: Prediction Result ──
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 9, "  Prediction Result", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 13)
    pdf.cell(90, 10, "Predicted Traffic Volume:", border=1)
    pdf.cell(100, 10, str(r.get('traffic_volume', 'N/A')), border=1, ln=True)

    pdf.set_font("Arial", 'B', 13)
    pdf.cell(90, 10, "Traffic Level:", border=1)
    pdf.cell(100, 10, level_text, border=1, ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "--- End of Report ---", ln=True, align="C")

    # ── Output ── (works for both fpdf v1.x and v2.x)
    raw = pdf.output(dest='S')
    if isinstance(raw, str):
        pdf_bytes = raw.encode('latin-1')
    else:
        pdf_bytes = bytes(raw)
    buffer = io.BytesIO(pdf_bytes)
    return send_file(buffer, as_attachment=True, download_name="traffic_result.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)