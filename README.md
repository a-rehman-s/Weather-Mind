# 🌦️ WeatherMind — AI-Powered Multi-City Weather Dashboard

WeatherMind is an intelligent weather forecasting dashboard that combines machine learning with real-time data to deliver accurate and interactive forecasts across multiple cities.

---

## 🚀 Features

- 🌍 Multi-city weather tracking  
- 📊 Interactive dashboard with modern UI  
- ⏱️ Hourly & 7-day forecasts  
- 🤖 Machine Learning-based prediction system  
- 📈 Forecast confidence metrics (MAE, RMSE, R²)  
- 🔁 City comparison mode  
- ⚡ Scalable backend with API-ready architecture  

---

## 🧠 Machine Learning

- Uses **Gradient Boosting Regressor** for temperature prediction  
- Trained on **simulated 5-year historical climate data**

### Feature Engineering

**Time-based features:**
- Month  
- Season  
- Day of year  

**Weather indicators:**
- Humidity  
- Wind speed  
- Pressure  
- UV index  

**Lag features:**
- Previous temperatures  
- Temperature trends

---

## 🛠️ Tech Stack

### Frontend
- HTML  
- CSS  
- Chart.js  

### Backend
- FastAPI  
- Python  

### Machine Learning
- Scikit-learn  
- Pandas  
- NumPy  

---

## ⚙️ How It Works

1. User selects or adds a city  
2. Backend generates or fetches weather data  
3. ML model predicts temperature trends  
4. Dashboard visualizes:
   - Current conditions  
   - Hourly forecasts  
   - Weekly forecasts  
   - Model confidence  

---

## 🧪 Current Status

- ✅ UI fully developed  
- ✅ Multi-city system implemented  
- ✅ Mock data integrated  
- ✅ ML pipeline structured on simulated data  
- 🔄 Backend integration in progress  
- 🔄 Real API integration upcoming  
