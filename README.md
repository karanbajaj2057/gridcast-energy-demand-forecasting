# ⚡ GridCast: AI-Powered Energy Consumption Forecasting

[![Live Demo](https://img.shields.io/badge/Live_Application-aithatpredictsfuturepower.lovable.app-00C7B7?style=for-the-badge&logo=vercel&logoColor=white)](https://aithatpredictsfuturepower.lovable.app/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Machine Learning](https://img.shields.io/badge/Model-Ridge_Regression-FF6F00?style=for-the-badge)](https://en.wikipedia.org/wiki/Ridge_regression)

> **Live Demo:** [https://aithatpredictsfuturepower.lovable.app/](https://aithatpredictsfuturepower.lovable.app/)

## 📌 Project Overview

**GridCast** is an interactive, client-side Machine Learning web application designed to model and forecast electrical grid load demand. 

Unlike static dashboards that display pre-computed outputs, GridCast executes **live in-browser Ridge Regression training** across **5,693 days (2003–2018)** of real historical electricity consumption data from the **PJM East Interconnection** (serving ~65 million people across the US Mid-Atlantic region). Users can dynamically adjust the train/test chronological split and customize the future forecast horizon to evaluate model generalization in real time.

---

## ✨ Key Features

* 🧠 **Live In-Browser ML Training:** Trains a regularized Ridge Regression ($L_2$) time-series model directly in the client browser with zero backend latency.
* 🎛️ **Interactive Hyperparameter & Split Controls:** 
  * Dynamically adjust the **Train / Test Split Date** to observe how historical training volume impacts unseen test performance.
  * Customize the **Future Forecast Horizon** slider from **7 days to 90 days** ahead.
* 📊 **Real-Time Baseline Benchmarking:** Automatically computes error metrics against a Naive Baseline model to prove predictive value and skill score.
* 📈 **Multi-Tab Analytical Dashboard:** Explore day-ahead forecasts on unseen data, residual error distributions, seasonal load patterns, and future projection curves.

---

## 📊 Model Performance & Evaluation Metrics

Using the default chronological validation split (Test period: `2017-01-01` to `2018-08-02` — **579 unseen days**), the Ridge Regression model significantly outperforms the Naive baseline across all standard regression metrics:

| Evaluation Metric | GridCast (Ridge Regression) | Naive Baseline | Performance Gain |
| :--- | :---: | :---: | :---: |
| **MAPE** *(Mean Absolute Percentage Error)* | **4.72%** | 10.16% | **-5.44%** error reduction |
| **RMSE** *(Root Mean Squared Error)* | **1,955 MW** | 4,326 MW | **-2,371 MW** error reduction |
| **MAE** *(Mean Absolute Error)* | **1,486 MW** | 3,229 MW | **-1,743 MW** error reduction |
| **R²** *(Coefficient of Determination)* | **0.813** | 0.085 | **+0.728** variance explained |

---

## 🔬 Dataset & Methodology

### 1. Dataset (`PJM East · 2003–2018`)
* **Source:** PJM Interconnection LLC (Eastern Region) historical load data.
* **Span:** 5,693 daily observations spanning 15+ years of seasonal, weekly, and macroeconomic demand shifts.
* **Target Variable:** Daily Megawatt (MW) electricity demand.

### 2. Modeling Approach (Ridge Regression)
Energy consumption exhibits strong autocorrelation (lag effects) and cyclical seasonality (summer cooling peaks and winter heating peaks). To prevent multicollinearity among lagged features and Fourier seasonal terms from causing overfitting, **Ridge Regression** minimizes the penalized residual sum of squares:

$$\min_{w} \| Xw - y \|_2^2 + \alpha \| w \|_2^2$$

Where $\alpha$ controls the $L_2$ regularization strength, ensuring stable multi-step projections across the 7-to-90-day forecast horizon.

---

## 🛠️ Tech Stack

* **Frontend Framework:** React 18 + TypeScript
* **Build Tool:** Vite
* **Styling & UI Components:** Tailwind CSS, Radix UI / Shadcn UI, Lucide Icons
* **Data Visualization:** Recharts (Interactive responsive time-series charting)
* **Machine Learning Engine:** Custom TypeScript matrix & linear algebra implementation for client-side Ridge Regression

---

## 🚀 Getting Started (Local Installation)

To run this project locally on your machine, follow these steps:

### Prerequisites
* **Node.js** (v18.0.0 or higher recommended)
* **npm** or **bun**

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/gridcast-energy-demand-forecasting.git
   ```

2. **Navigate into the project directory:**
   ```bash
   cd gridcast-energy-demand-forecasting
   ```

3. **Install dependencies:**
   ```bash
   npm install
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Open in Browser:**
   Visit `http://localhost:5173` to view the live application.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
