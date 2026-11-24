# 🎓 Student Exam Performance — Analytics & ML Dashboard

> **Author:** Kumkum  
> **Dataset:** https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset
> **Stack:** Python · Streamlit · Pandas · NumPy · Plotly · Scikit-Learn

---

## 📌 Project Overview

This project is an end-to-end **interactive analytics and machine learning dashboard** built entirely in Python. It explores what factors drive student exam performance — covering demographics, study habits, lifestyle, and technology access — and uses **Random Forest** models to predict both pass/fail outcome and exact exam score for any hypothetical student profile.

---

## 🗂️ Repository Structure

```
IBM Project/
│
├── student_performance_app.py       ← ✅ Single-file Streamlit app  (run this)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
├── student_exam_performance.csv     ← Dataset (must be in the same folder)
│
└── student_performance_app/         ← Modular version (same logic, split across files)
    ├── app.py                       ← Entry-point
    ├── backend/
    │   ├── data_loader.py           ← Data loading & feature engineering
    │   ├── analytics.py             ← All Plotly chart functions
    │   └── ml_model.py              ← ML training, metrics, prediction
    └── frontend/
        └── pages/
            ├── overview.py
            ├── demographics.py
            ├── study_habits.py
            ├── performance_analysis.py
            └── ml_prediction.py
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd "IBM Project"
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

**Single-file version (recommended):**
```bash
streamlit run student_performance_app.py
```

**Modular version:**
```bash
streamlit run student_performance_app/app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 📊 Dashboard Pages

### 1. 📊 Overview
High-level snapshot of the entire dataset.

| Widget | Description |
|--------|-------------|
| KPI cards | Total students, pass rate, avg score, avg attendance, avg study hours |
| Grade donut chart | Distribution across A / B / C / D / F |
| Score histogram | Exam score frequency coloured by grade |
| Pass/Fail bar | Count of students who passed vs failed |
| Box plot | Score spread within each grade band |
| Descriptive stats table | Min, max, mean, std, quartiles for key numeric columns |

---

### 2. 👥 Demographics
How student background affects performance.

| Chart | Insight |
|-------|---------|
| Violin plot | Score distribution by gender |
| Scatter | Age vs exam score coloured by grade |
| Bar | Average score by education level |
| Bar | Average score by family income bracket |
| Bar | Average score by urban / rural location |
| Donut | School type breakdown (Public / Private / Charter) |
| Summary table | Pass rate + avg score by any demographic group |

---

### 3. 📚 Study Habits
How studying, sleep, and mental state affect outcomes.

| Chart | Insight |
|-------|---------|
| Scatter + trendline | Study hours per day vs exam score |
| Horizontal bar | Average score by study method (Flashcards / Group Study / etc.) |
| Scatter + trendline | Attendance % vs exam score |
| Bar | Sleep quality vs average score |
| Heatmap | Stress level × motivation level → average score |
| Bar | Average score by study environment (Quiet / Moderate / Noisy) |
| Summary table | Per-method stats: students, avg score, study hrs, attendance, pass rate |

---

### 4. 📈 Performance Analysis
Deep-dive correlations and exam-specific factors.

| Chart | Insight |
|-------|---------|
| Horizontal bar | Top 15 features correlated with exam score (Pearson r) |
| Scatter + trendline | Previous GPA vs exam score |
| Bar with error bars | Average score by exam difficulty (Easy / Medium / Hard) |
| Scatter + trendline | Number of preparation days vs exam score |
| Heatmap | Full feature correlation matrix |
| Percentile explorer | How any given score ranks in the filtered dataset |
| Grade metrics table | Average score, GPA, study hours, attendance, prep days per grade |

---

### 5. 🤖 ML Prediction

#### Model Metrics tab
Two **Random Forest** models trained on 80% of the 100k dataset:

| Model | Target | Metric |
|-------|--------|--------|
| Classifier | Pass (1) / Fail (0) | Accuracy, Precision, Recall, F1 |
| Regressor | Exam Score (0–100) | MAE, R² |

Visualisations: confusion matrix, feature importance (top 20), actual vs predicted scatter.

#### Predict a Student tab
Fill in **25 sliders** (academic history, study habits, sleep, stress, motivation…) and click **Predict** to get:
- Predicted exam score
- Predicted grade (A–F)
- Pass probability with a colour-coded progress bar

---

## 🔍 Sidebar Filters

Every page respects **global sidebar filters** applied in real time:

| Filter | Options |
|--------|---------|
| Gender | All / Female / Male / Other |
| Education Level | All / High School / Undergraduate / … |
| School Type | All / Public / Private / Charter |
| Family Income | All / Low → High |
| Exam Score Range | 0–100 slider |

> **Note:** ML models always train on the **full unfiltered dataset** to ensure reliability; the filters only affect the analytics pages.

---

## 🤖 Machine Learning Details

### Features used (25 inputs)
| Category | Features |
|----------|---------|
| Academic | previous_exam_score, previous_gpa, attendance, assignment_completion_rate |
| Study | study_hours_per_day, self_study_hours, online_learning_hours, practice_tests_completed |
| Lifestyle | sleep_hours, daily_screen_time, physical_activity_hours |
| Technology | online_course_hours |
| Exam | exam_preparation_days, questions_attempted, time_management_score, exam_anxiety_level |
| Ordinal encoded | stress, motivation, study_consistency, sleep_quality, revision_frequency, class_participation, notes_quality, educational_app_usage |

### Model parameters
```python
RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
RandomForestRegressor(n_estimators=200,  max_depth=12, random_state=42, n_jobs=-1)
```

Models are cached with `@st.cache_resource` — trained once per session, reused instantly on page navigation.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.35 | Web UI framework |
| `pandas` | ≥ 2.0 | Data wrangling |
| `numpy` | ≥ 1.24 | Numeric operations |
| `plotly` | ≥ 5.18 | Interactive charts |
| `scikit-learn` | ≥ 1.3 | Random Forest models |
| `statsmodels` | ≥ 0.14 | OLS trendlines in scatter plots |
| `pyarrow` | ≥ 14 | Fast CSV reads via pandas |

---

## 📋 Dataset Description

| Attribute | Value |
|-----------|-------|
| File | `student_exam_performance.csv` |
| Rows | 100,000 |
| Columns | 44 |
| Target columns | `exam_score`, `performance_grade`, `pass_status`, `performance_level` |

**Feature categories:**
- **Demographics** — age, gender, education level, school type, family income, parent education, urban/rural
- **Academic history** — previous exam score, previous GPA, attendance %, assignment completion, class participation
- **Study behaviour** — study hours, self-study hours, study method, consistency, environment, revision frequency, practice tests, notes quality, private tuition
- **Lifestyle** — sleep hours, sleep quality, screen time, physical activity, break frequency, stress, motivation
- **Technology** — internet access, device availability, educational app usage, online course hours
- **Exam context** — exam difficulty, preparation days, questions attempted/correct, time management score, anxiety level

---

## 🚀 Quick Start (TL;DR)

```bash
pip install -r requirements.txt
streamlit run student_performance_app.py
```

Open **http://localhost:8501** in your browser. Done. ✅
