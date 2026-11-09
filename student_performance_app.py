"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         STUDENT EXAM PERFORMANCE — ANALYTICS & ML DASHBOARD                  ║
║         Single-file Streamlit application                                    ║
║         Dataset : https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset ║
║         Run     : streamlit run student_performance_app.py                   ║
║         Author  : Kumkum                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONSTANTS & COLUMN GROUPS
# ─────────────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_HERE, "student_exam_performance.csv")

NUMERIC_COLS = [
    "age", "previous_exam_score", "previous_gpa",
    "attendance_percentage", "assignment_completion_rate",
    "study_hours_per_day", "self_study_hours", "online_learning_hours",
    "practice_tests_completed", "sleep_hours", "daily_screen_time",
    "physical_activity_hours", "online_course_hours",
    "exam_preparation_days", "questions_attempted", "questions_correct",
    "time_management_score", "exam_anxiety_level", "exam_score",
]

ORDINAL_MAPS = {
    "stress_level":          {"Low": 1, "Medium": 2, "High": 3},
    "motivation_level":      {"Low": 1, "Medium": 2, "High": 3},
    "class_participation":   {"Low": 1, "Medium": 2, "High": 3},
    "study_consistency":     {"Low": 1, "Medium": 2, "High": 3},
    "sleep_quality":         {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4},
    "notes_quality":         {"Poor": 1, "Average": 2, "Excellent": 3},
    "educational_app_usage": {"Low": 1, "Moderate": 2, "High": 3},
    "revision_frequency":    {"Rarely": 1, "Weekly": 2, "Daily": 3},
    "break_frequency":       {"Rarely": 1, "Occasionally": 2, "Frequently": 3},
    "performance_level":     {"Low": 1, "Medium": 2, "High": 3},
    "performance_grade":     {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4},
}

GRADE_COLORS = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
_TEMPLATE   = "plotly_white"
_PASTEL     = px.colors.qualitative.Pastel

CLASSIFIER_FEATURES = [
    "age", "previous_exam_score", "previous_gpa",
    "attendance_percentage", "assignment_completion_rate",
    "study_hours_per_day", "self_study_hours", "online_learning_hours",
    "practice_tests_completed", "sleep_hours", "daily_screen_time",
    "physical_activity_hours", "online_course_hours",
    "exam_preparation_days", "questions_attempted",
    "time_management_score", "exam_anxiety_level",
    "stress_level_num", "motivation_level_num", "study_consistency_num",
    "sleep_quality_num", "revision_frequency_num", "class_participation_num",
    "notes_quality_num", "educational_app_usage_num",
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load, clean, and enrich the student performance CSV."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Numeric coercion
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordinal encoding → new _num columns
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(mapping)

    # Binary pass flag
    if "pass_status" in df.columns:
        df["passed"] = (df["pass_status"] == "Pass").astype(int)

    if "private_tuition" in df.columns:
        df["private_tuition"] = pd.to_numeric(df["private_tuition"], errors="coerce")
    if "internet_access" in df.columns:
        df["internet_access"] = pd.to_numeric(df["internet_access"], errors="coerce")

    # Derived features
    if {"questions_correct", "questions_attempted"}.issubset(df.columns):
        df["accuracy_rate"] = np.where(
            df["questions_attempted"] > 0,
            (df["questions_correct"] / df["questions_attempted"] * 100).round(2),
            np.nan,
        )
    if {"study_hours_per_day", "self_study_hours"}.issubset(df.columns):
        df["total_study_hours"] = df["study_hours_per_day"] + df["self_study_hours"]

    return df


@st.cache_data(show_spinner=False)
def get_summary_stats(df: pd.DataFrame) -> dict:
    """Headline KPI values for the Overview page."""
    return {
        "total_students":  len(df),
        "pass_rate":       round(df["passed"].mean() * 100, 1) if "passed" in df.columns else None,
        "avg_score":       round(df["exam_score"].mean(), 2)   if "exam_score" in df.columns else None,
        "avg_attendance":  round(df["attendance_percentage"].mean(), 2) if "attendance_percentage" in df.columns else None,
        "avg_study_hours": round(df["study_hours_per_day"].mean(), 2)  if "study_hours_per_day" in df.columns else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHART FUNCTIONS  (Analytics)
# ─────────────────────────────────────────────────────────────────────────────

# ── Overview ──────────────────────────────────────────────────────────────────
def fig_grade_donut(df):
    counts = df["performance_grade"].value_counts().reindex(["A","B","C","D","F"]).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker_colors=[GRADE_COLORS.get(g,"#999") for g in counts.index],
        textinfo="label+percent",
        hovertemplate="<b>Grade %{label}</b><br>Count: %{value:,}<extra></extra>",
    ))
    fig.update_layout(template=_TEMPLATE, title="Grade Distribution",
                      legend=dict(orientation="h", y=-0.1), margin=dict(t=50,b=20))
    return fig

def fig_score_histogram(df):
    fig = px.histogram(
        df.dropna(subset=["exam_score","performance_grade"]),
        x="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, nbins=50, template=_TEMPLATE,
        labels={"exam_score":"Exam Score","performance_grade":"Grade"},
        title="Exam Score Distribution",
        category_orders={"performance_grade":["A","B","C","D","F"]},
    )
    fig.update_layout(bargap=0.05, legend_title="Grade")
    return fig

def fig_pass_fail_bar(df):
    counts = df["pass_status"].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=["#2ecc71" if s=="Pass" else "#e74c3c" for s in counts.index],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(template=_TEMPLATE, title="Pass / Fail Count",
                      xaxis_title="Status", yaxis_title="Students")
    return fig

def fig_score_box_by_grade(df):
    fig = px.box(
        df.dropna(subset=["exam_score","performance_grade"]),
        x="performance_grade", y="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, template=_TEMPLATE,
        title="Score Range by Grade",
        labels={"exam_score":"Exam Score","performance_grade":"Grade"},
        category_orders={"performance_grade":["A","B","C","D","F"]},
    )
    fig.update_layout(showlegend=False)
    return fig

# ── Demographics ──────────────────────────────────────────────────────────────
def fig_score_by_gender(df):
    fig = px.violin(df.dropna(subset=["exam_score","gender"]),
        x="gender", y="exam_score", color="gender", box=True, points=False,
        template=_TEMPLATE, title="Exam Score by Gender",
        labels={"exam_score":"Exam Score","gender":"Gender"},
        color_discrete_sequence=_PASTEL)
    fig.update_layout(showlegend=False)
    return fig

def fig_age_vs_score(df):
    fig = px.scatter(df.dropna(subset=["age","exam_score"]),
        x="age", y="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, opacity=0.35, template=_TEMPLATE,
        title="Age vs Exam Score",
        labels={"exam_score":"Exam Score","age":"Age"},
        category_orders={"performance_grade":["A","B","C","D","F"]})
    return fig

def fig_score_by_education(df):
    avg = df.groupby("education_level")["exam_score"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="education_level", y="exam_score", color="education_level",
        template=_TEMPLATE, title="Average Score by Education Level",
        labels={"exam_score":"Avg Score","education_level":"Level"},
        text_auto=".1f", color_discrete_sequence=_PASTEL)
    fig.update_layout(showlegend=False, xaxis_title="")
    return fig

def fig_score_by_income(df):
    order = ["Low","Lower-Middle","Middle","Upper-Middle","High"]
    present = [o for o in order if o in df["family_income"].unique()]
    avg = df.groupby("family_income")["exam_score"].mean().reindex(present).reset_index()
    fig = px.bar(avg, x="family_income", y="exam_score", color="family_income",
        template=_TEMPLATE, title="Average Score by Family Income",
        labels={"exam_score":"Avg Score","family_income":"Income"},
        text_auto=".1f", color_discrete_sequence=_PASTEL)
    fig.update_layout(showlegend=False, xaxis_title="")
    return fig

def fig_score_by_location(df):
    avg = df.groupby("urban_rural")["exam_score"].mean().reset_index()
    fig = px.bar(avg, x="urban_rural", y="exam_score", color="urban_rural",
        template=_TEMPLATE, title="Average Score by Location",
        labels={"exam_score":"Avg Score","urban_rural":"Location"},
        text_auto=".1f", color_discrete_sequence=_PASTEL)
    fig.update_layout(showlegend=False, xaxis_title="")
    return fig

def fig_school_type_donut(df):
    counts = df["school_type"].value_counts()
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.5,
        textinfo="label+percent", marker_colors=px.colors.qualitative.Set2))
    fig.update_layout(template=_TEMPLATE, title="School Type Distribution", margin=dict(t=50,b=20))
    return fig

# ── Study Habits ──────────────────────────────────────────────────────────────
def fig_study_hours_vs_score(df):
    fig = px.scatter(df.dropna(subset=["study_hours_per_day","exam_score"]),
        x="study_hours_per_day", y="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, opacity=0.3,
        trendline="ols", trendline_scope="overall", trendline_color_override="#e74c3c",
        template=_TEMPLATE, title="Study Hours per Day vs Exam Score",
        labels={"study_hours_per_day":"Study Hrs/Day","exam_score":"Exam Score"},
        category_orders={"performance_grade":["A","B","C","D","F"]})
    return fig

def fig_study_method_avg(df):
    avg = df.groupby("study_method")["exam_score"].mean().sort_values(ascending=True).reset_index()
    fig = px.bar(avg, x="exam_score", y="study_method", orientation="h",
        template=_TEMPLATE, title="Average Score by Study Method",
        labels={"exam_score":"Avg Score","study_method":"Method"},
        color="exam_score", color_continuous_scale="Blues", text_auto=".1f")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig

def fig_sleep_vs_score(df):
    avg = df.groupby("sleep_quality")["exam_score"].mean().reset_index()
    order = ["Poor","Fair","Good","Excellent"]
    avg["sleep_quality"] = pd.Categorical(avg["sleep_quality"], categories=order, ordered=True)
    avg = avg.sort_values("sleep_quality")
    fig = px.bar(avg, x="sleep_quality", y="exam_score",
        template=_TEMPLATE, title="Sleep Quality vs Average Score",
        labels={"exam_score":"Avg Score","sleep_quality":"Sleep Quality"},
        color="exam_score", color_continuous_scale="Greens", text_auto=".1f")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig

def fig_attendance_vs_score(df):
    fig = px.scatter(df.dropna(subset=["attendance_percentage","exam_score"]),
        x="attendance_percentage", y="exam_score", color="pass_status",
        color_discrete_map={"Pass":"#2ecc71","Fail":"#e74c3c"}, opacity=0.25,
        trendline="ols", trendline_scope="overall", trendline_color_override="#2c3e50",
        template=_TEMPLATE, title="Attendance % vs Exam Score",
        labels={"attendance_percentage":"Attendance %","exam_score":"Exam Score"})
    return fig

def fig_stress_motivation_heatmap(df):
    pivot = (df.dropna(subset=["stress_level","motivation_level"])
        .groupby(["stress_level","motivation_level"])["exam_score"].mean().unstack())
    row_order = ["Low","Medium","High"]
    pivot = pivot.reindex(index=row_order, columns=row_order)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn", text=np.round(pivot.values, 1), texttemplate="%{text}",
        hovertemplate="Stress: %{y}<br>Motivation: %{x}<br>Avg: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(template=_TEMPLATE, title="Avg Score: Stress × Motivation",
                      xaxis_title="Motivation Level", yaxis_title="Stress Level")
    return fig

def fig_study_environment_avg(df):
    avg = df.groupby("study_environment")["exam_score"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="study_environment", y="exam_score",
        template=_TEMPLATE, title="Average Score by Study Environment",
        labels={"exam_score":"Avg Score","study_environment":"Environment"},
        color="exam_score", color_continuous_scale="Purples", text_auto=".1f")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig

# ── Performance Analysis ──────────────────────────────────────────────────────
def fig_top_correlations(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if "student_id" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["student_id"])
    display_cols = [c for c in numeric_df.columns if c != "exam_score"]
    corr = numeric_df[display_cols + ["exam_score"]].corr()["exam_score"].drop("exam_score")
    corr = corr.dropna().sort_values(key=abs, ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in corr.values],
        text=[f"{v:.2f}" for v in corr.values], textposition="outside",
    ))
    fig.update_layout(template=_TEMPLATE, title="Top Feature Correlations with Exam Score",
                      xaxis_title="Pearson r", height=480, margin=dict(l=160))
    return fig

def fig_correlation_heatmap(df, cols):
    sub = df[cols].dropna()
    corr = sub.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(template=_TEMPLATE, title="Feature Correlation Matrix",
                      height=550, xaxis=dict(tickangle=-35))
    return fig

def fig_gpa_vs_score(df):
    fig = px.scatter(df.dropna(subset=["previous_gpa","exam_score"]),
        x="previous_gpa", y="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, opacity=0.3,
        trendline="ols", trendline_scope="overall", trendline_color_override="#2c3e50",
        template=_TEMPLATE, title="Previous GPA vs Exam Score",
        labels={"previous_gpa":"Previous GPA","exam_score":"Exam Score"},
        category_orders={"performance_grade":["A","B","C","D","F"]})
    return fig

def fig_exam_difficulty_impact(df):
    avg = (df.dropna(subset=["exam_difficulty","exam_score"])
        .groupby("exam_difficulty")["exam_score"].agg(["mean","std"]).reset_index())
    order = ["Easy","Medium","Hard"]
    avg["exam_difficulty"] = pd.Categorical(avg["exam_difficulty"], categories=order, ordered=True)
    avg = avg.sort_values("exam_difficulty")
    fig = go.Figure(go.Bar(
        x=avg["exam_difficulty"], y=avg["mean"],
        error_y=dict(type="data", array=avg["std"].values, visible=True),
        marker_color=["#2ecc71","#f39c12","#e74c3c"],
        text=avg["mean"].round(1), textposition="outside",
    ))
    fig.update_layout(template=_TEMPLATE, title="Average Score by Exam Difficulty (± Std Dev)",
                      xaxis_title="Difficulty", yaxis_title="Avg Score")
    return fig

def fig_preparation_days_vs_score(df):
    fig = px.scatter(df.dropna(subset=["exam_preparation_days","exam_score"]),
        x="exam_preparation_days", y="exam_score", color="performance_grade",
        color_discrete_map=GRADE_COLORS, opacity=0.3,
        trendline="ols", trendline_scope="overall", trendline_color_override="#8e44ad",
        template=_TEMPLATE, title="Exam Preparation Days vs Score",
        labels={"exam_preparation_days":"Prep Days","exam_score":"Exam Score"},
        category_orders={"performance_grade":["A","B","C","D","F"]})
    return fig

# ── ML Charts ────────────────────────────────────────────────────────────────
def fig_feature_importance(model, feature_names, title="Feature Importance"):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(20)
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=px.colors.sequential.Blues_r[:len(imp)],
        text=[f"{v:.3f}" for v in imp.values], textposition="outside",
    ))
    fig.update_layout(template="plotly_white", title=title,
                      xaxis_title="Importance", height=500, margin=dict(l=160))
    return fig

def fig_confusion_matrix(conf_mat):
    labels = ["Fail","Pass"]
    fig = go.Figure(go.Heatmap(
        z=conf_mat, x=labels, y=labels, colorscale="Blues",
        text=conf_mat, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(template="plotly_white", title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def fig_actual_vs_predicted(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
        marker=dict(color="#3498db", opacity=0.3, size=4), name="Predictions"))
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=2), name="Perfect Fit"))
    fig.update_layout(template="plotly_white", title="Actual vs Predicted Exam Score",
                      xaxis_title="Actual Score", yaxis_title="Predicted Score")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _prepare(df, features, target):
    cols = features + [target]
    present = [c for c in cols if c in df.columns]
    sub = df[present].dropna()
    X = sub[[c for c in features if c in sub.columns]]
    y = sub[target]
    return X, y

@st.cache_resource(show_spinner="Training pass/fail classifier…")
def train_classifier(df: pd.DataFrame):
    X, y = _prepare(df, CLASSIFIER_FEATURES, "passed")
    feat_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, feat_cols

@st.cache_resource(show_spinner="Training exam score regressor…")
def train_regressor(df: pd.DataFrame):
    X, y = _prepare(df, CLASSIFIER_FEATURES, "exam_score")
    feat_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    return reg, X_test, y_test, feat_cols

def predict_student(clf, reg, feat_cols, inputs):
    row = pd.DataFrame([{c: inputs.get(c, np.nan) for c in feat_cols}])
    pass_prob  = clf.predict_proba(row)[0][1] * 100
    pred_score = reg.predict(row)[0]
    return round(pass_prob, 1), round(pred_score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Page 1 : Overview ─────────────────────────────────────────────────────────
def page_overview(df, df_full):
    st.title("📊 Overview Dashboard")
    st.caption(f"Showing **{len(df):,}** students (filtered) out of **{len(df_full):,}** total")

    stats = get_summary_stats(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",    f"{stats['total_students']:,}")
    c2.metric("Pass Rate",         f"{stats['pass_rate']}%")
    c3.metric("Avg Exam Score",    f"{stats['avg_score']}")
    c4.metric("Avg Attendance",    f"{stats['avg_attendance']}%")
    c5.metric("Avg Study Hrs/Day", f"{stats['avg_study_hours']} hrs")

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    col1.plotly_chart(fig_grade_donut(df),      use_container_width=True)
    col2.plotly_chart(fig_score_histogram(df),  use_container_width=True)

    col3, col4 = st.columns([1, 2])
    col3.plotly_chart(fig_pass_fail_bar(df),       use_container_width=True)
    col4.plotly_chart(fig_score_box_by_grade(df),  use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Descriptive Statistics")
    num_cols = ["exam_score","previous_exam_score","previous_gpa",
                "attendance_percentage","study_hours_per_day","sleep_hours","exam_preparation_days"]
    desc = df[[c for c in num_cols if c in df.columns]].describe().T.round(2)
    desc.columns = [c.title() for c in desc.columns]
    st.dataframe(desc, use_container_width=True)


# ── Page 2 : Demographics ─────────────────────────────────────────────────────
def page_demographics(df, df_full):
    st.title("👥 Demographics Analysis")
    st.caption(f"Showing **{len(df):,}** students")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_score_by_gender(df),    use_container_width=True)
    col2.plotly_chart(fig_age_vs_score(df),        use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_score_by_education(df),  use_container_width=True)
    col4.plotly_chart(fig_score_by_income(df),     use_container_width=True)

    col5, col6 = st.columns(2)
    col5.plotly_chart(fig_score_by_location(df),   use_container_width=True)
    col6.plotly_chart(fig_school_type_donut(df),   use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Pass Rate by Group")
    group_sel = st.selectbox("Group by", ["gender","education_level","school_type","family_income","urban_rural"])
    summary = (df.dropna(subset=[group_sel,"passed"]).groupby(group_sel)
        .agg(Students=("student_id","count"), Avg_Score=("exam_score","mean"), Pass_Rate=("passed","mean"))
        .rename(columns={"Avg_Score":"Avg Score","Pass_Rate":"Pass Rate"})
        .assign(**{"Avg Score": lambda d: d["Avg Score"].round(2),
                   "Pass Rate": lambda d: (d["Pass Rate"]*100).round(1).astype(str)+"%"})
        .sort_values("Students", ascending=False))
    st.dataframe(summary, use_container_width=True)


# ── Page 3 : Study Habits ─────────────────────────────────────────────────────
def page_study_habits(df, df_full):
    st.title("📚 Study Habits Analysis")
    st.caption(f"Showing **{len(df):,}** students")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_study_hours_vs_score(df), use_container_width=True)
    col2.plotly_chart(fig_study_method_avg(df),     use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_attendance_vs_score(df),  use_container_width=True)
    col4.plotly_chart(fig_sleep_vs_score(df),       use_container_width=True)

    col5, col6 = st.columns(2)
    col5.plotly_chart(fig_stress_motivation_heatmap(df),  use_container_width=True)
    col6.plotly_chart(fig_study_environment_avg(df),      use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Study Behaviour Stats by Method")
    if "study_method" in df.columns:
        tbl = (df.dropna(subset=["study_method"]).groupby("study_method")
            .agg(Students=("student_id","count"), Avg_Score=("exam_score","mean"),
                 Avg_Study_Hrs=("study_hours_per_day","mean"),
                 Avg_Attendance=("attendance_percentage","mean"), Pass_Rate=("passed","mean"))
            .rename(columns={"Avg_Score":"Avg Score","Avg_Study_Hrs":"Study Hrs/Day",
                              "Avg_Attendance":"Attendance %","Pass_Rate":"Pass Rate"})
            .assign(**{"Avg Score":     lambda d: d["Avg Score"].round(2),
                       "Study Hrs/Day": lambda d: d["Study Hrs/Day"].round(2),
                       "Attendance %":  lambda d: d["Attendance %"].round(1),
                       "Pass Rate":     lambda d: (d["Pass Rate"]*100).round(1).astype(str)+"%"})
            .sort_values("Avg Score", ascending=False))
        st.dataframe(tbl, use_container_width=True)


# ── Page 4 : Performance Analysis ────────────────────────────────────────────
_CORR_FEATURES = [
    "exam_score","previous_exam_score","previous_gpa","attendance_percentage",
    "study_hours_per_day","exam_preparation_days","time_management_score",
    "exam_anxiety_level","sleep_hours","stress_level_num","motivation_level_num",
    "study_consistency_num","practice_tests_completed",
]

def page_performance_analysis(df, df_full):
    st.title("📈 Performance Analysis")
    st.caption(f"Showing **{len(df):,}** students")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_top_correlations(df),       use_container_width=True)
    col2.plotly_chart(fig_gpa_vs_score(df),            use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_exam_difficulty_impact(df),  use_container_width=True)
    col4.plotly_chart(fig_preparation_days_vs_score(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Feature Correlation Heatmap")
    present = [c for c in _CORR_FEATURES if c in df.columns]
    if len(present) >= 3:
        st.plotly_chart(fig_correlation_heatmap(df, present), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Score Percentile Explorer")
    score_input = st.slider("Select an exam score", 0, 100, 75)
    pct = round((df["exam_score"].dropna() < score_input).mean() * 100, 1)
    st.info(f"A score of **{score_input}** is better than **{pct}%** of students in the current filter.")

    st.markdown("---")
    st.markdown("### 📋 Grade vs Key Metrics")
    if "performance_grade" in df.columns:
        tbl = (df.dropna(subset=["performance_grade"]).groupby("performance_grade")
            .agg(Count=("student_id","count"), Avg_Score=("exam_score","mean"),
                 Avg_GPA=("previous_gpa","mean"), Study_Hrs=("study_hours_per_day","mean"),
                 Attendance=("attendance_percentage","mean"), Prep_Days=("exam_preparation_days","mean"))
            .rename(columns={"Avg_Score":"Avg Score","Avg_GPA":"Avg GPA",
                              "Study_Hrs":"Study Hrs/Day","Attendance":"Attendance %","Prep_Days":"Prep Days"})
            .round(2).reindex(["A","B","C","D","F"]).dropna(how="all"))
        st.dataframe(tbl, use_container_width=True)


# ── Page 5 : ML Prediction ────────────────────────────────────────────────────
_ORDINAL_INPUTS = {
    "stress_level_num":          ("Stress Level",          1, 3, 2, {1:"Low",2:"Medium",3:"High"}),
    "motivation_level_num":      ("Motivation Level",      1, 3, 2, {1:"Low",2:"Medium",3:"High"}),
    "study_consistency_num":     ("Study Consistency",     1, 3, 2, {1:"Low",2:"Medium",3:"High"}),
    "sleep_quality_num":         ("Sleep Quality",         1, 4, 3, {1:"Poor",2:"Fair",3:"Good",4:"Excellent"}),
    "revision_frequency_num":    ("Revision Frequency",    1, 3, 2, {1:"Rarely",2:"Weekly",3:"Daily"}),
    "class_participation_num":   ("Class Participation",   1, 3, 2, {1:"Low",2:"Medium",3:"High"}),
    "notes_quality_num":         ("Notes Quality",         1, 3, 2, {1:"Poor",2:"Average",3:"Excellent"}),
    "educational_app_usage_num": ("Educational App Usage", 1, 3, 2, {1:"Low",2:"Moderate",3:"High"}),
}
_NUMERIC_INPUTS = {
    "age":                       ("Age",                      15, 25, 18, 1),
    "previous_exam_score":       ("Previous Exam Score",      0, 100, 70, 1),
    "previous_gpa":              ("Previous GPA",             0.0, 4.0, 3.0, 0.1),
    "attendance_percentage":     ("Attendance %",             0, 100, 85, 1),
    "assignment_completion_rate":("Assignment Completion %",  0, 100, 75, 1),
    "study_hours_per_day":       ("Study Hours/Day",          0.0, 12.0, 4.0, 0.5),
    "self_study_hours":          ("Self Study Hours/Day",     0.0, 8.0,  2.0, 0.5),
    "online_learning_hours":     ("Online Learning Hrs",      0.0, 10.0, 2.0, 0.5),
    "practice_tests_completed":  ("Practice Tests Done",      0, 20, 5, 1),
    "sleep_hours":               ("Sleep Hours/Night",        4.0, 12.0, 7.0, 0.5),
    "daily_screen_time":         ("Daily Screen Time (hrs)",  0.0, 12.0, 4.0, 0.5),
    "physical_activity_hours":   ("Physical Activity Hrs",    0.0, 6.0,  1.0, 0.5),
    "online_course_hours":       ("Online Course Hrs",        0.0, 10.0, 2.0, 0.5),
    "exam_preparation_days":     ("Exam Preparation Days",    0, 60, 14, 1),
    "questions_attempted":       ("Questions Attempted",      0, 100, 90, 1),
    "time_management_score":     ("Time Management Score",    0.0, 10.0, 7.0, 0.5),
    "exam_anxiety_level":        ("Exam Anxiety Level",       0.0, 10.0, 5.0, 0.5),
}

def page_ml_prediction(df, df_full):
    st.title("🤖 ML Prediction & Model Insights")
    tab_metrics, tab_predict = st.tabs(["📊 Model Metrics", "🔮 Predict a Student"])

    clf, X_tc, y_tc, fc_c = train_classifier(df_full)
    reg, X_tr, y_tr, fc_r = train_regressor(df_full)

    with tab_metrics:
        y_pred_c = clf.predict(X_tc)
        report   = classification_report(y_tc, y_pred_c, target_names=["Fail","Pass"], output_dict=True)
        cm       = confusion_matrix(y_tc, y_pred_c)
        acc      = round(accuracy_score(y_tc, y_pred_c) * 100, 2)

        st.subheader("Classifier — Pass / Fail Prediction (Random Forest)")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy",         f"{acc}%")
        m2.metric("Precision (Pass)", f"{report['Pass']['precision']*100:.1f}%")
        m3.metric("Recall (Pass)",    f"{report['Pass']['recall']*100:.1f}%")
        m4.metric("F1 (Pass)",        f"{report['Pass']['f1-score']*100:.1f}%")

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_confusion_matrix(cm), use_container_width=True)
        col2.plotly_chart(fig_feature_importance(clf, fc_c, "Pass/Fail — Feature Importance"), use_container_width=True)

        st.markdown("---")
        st.subheader("Regressor — Exam Score Prediction (Random Forest)")
        y_pred_r = reg.predict(X_tr)
        mae = round(mean_absolute_error(y_tr, y_pred_r), 2)
        r2  = round(r2_score(y_tr, y_pred_r), 4)
        r1, r2_col = st.columns(2)
        r1.metric("Mean Absolute Error (MAE)", f"{mae} pts")
        r2_col.metric("R² Score", f"{r2}")

        col3, col4 = st.columns(2)
        col3.plotly_chart(fig_actual_vs_predicted(y_tr, y_pred_r), use_container_width=True)
        col4.plotly_chart(fig_feature_importance(reg, fc_r, "Score — Feature Importance"), use_container_width=True)

    with tab_predict:
        st.subheader("🔮 Enter a Student Profile")
        st.info("Adjust sliders and hit **Predict** to get the model's forecast.")
        inputs = {}

        num_keys = list(_NUMERIC_INPUTS.keys())
        cols = st.columns(3)
        for i, key in enumerate(num_keys):
            label, mn, mx, default, step = _NUMERIC_INPUTS[key]
            with cols[i % 3]:
                inputs[key] = st.slider(label,
                    float(mn) if isinstance(step, float) else int(mn),
                    float(mx) if isinstance(step, float) else int(mx),
                    float(default) if isinstance(step, float) else int(default),
                    step)

        st.markdown("#### Qualitative Factors")
        ord_keys = list(_ORDINAL_INPUTS.keys())
        ord_cols = st.columns(4)
        for i, key in enumerate(ord_keys):
            label, mn, mx, default, mapping = _ORDINAL_INPUTS[key]
            with ord_cols[i % 4]:
                val = st.selectbox(label, options=list(mapping.keys()),
                                   index=default - 1, format_func=lambda x, m=mapping: m[x])
                inputs[key] = val

        st.markdown("---")
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            pass_prob, pred_score = predict_student(clf, reg, fc_c, inputs)
            pred_grade = ("A" if pred_score >= 90 else "B" if pred_score >= 75 else
                          "C" if pred_score >= 60 else "D" if pred_score >= 50 else "F")

            res1, res2, res3 = st.columns(3)
            res1.metric("Predicted Exam Score", str(pred_score))
            res2.metric("Predicted Grade",      pred_grade)
            res3.metric("Pass Probability",     f"{pass_prob}%")

            color = "#2ecc71" if pass_prob >= 60 else "#f39c12" if pass_prob >= 40 else "#e74c3c"
            st.markdown(f"""
            <div style="background:{color}22;border:2px solid {color};border-radius:10px;
                        padding:16px 24px;margin-top:12px;">
              <div style="font-size:1.1rem;font-weight:600;color:{color};">
                {"✅ Likely to PASS" if pass_prob >= 50 else "⚠️ At Risk of FAILING"}
                &nbsp;—&nbsp; {pass_prob}% pass probability
              </div>
              <div style="background:#e5e7eb;border-radius:6px;height:14px;margin-top:10px;">
                <div style="background:{color};width:{pass_prob}%;height:14px;border-radius:6px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STREAMLIT APP ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Student Exam Performance Dashboard",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        [data-testid="stSidebar"] { background: #1a1a2e; }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }
    </style>""", unsafe_allow_html=True)

    # Load data
    df_full = load_data()

    # Sidebar — navigation & filters
    with st.sidebar:
        st.markdown("## 🎓 Student Performance")
        st.markdown("---")

        page = st.radio("Navigate", [
            "📊 Overview",
            "👥 Demographics",
            "📚 Study Habits",
            "📈 Performance Analysis",
            "🤖 ML Prediction",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        sel_gender = st.selectbox("Gender",
            ["All"] + sorted(df_full["gender"].dropna().unique().tolist()))
        sel_edu = st.selectbox("Education Level",
            ["All"] + sorted(df_full["education_level"].dropna().unique().tolist()))
        sel_school = st.selectbox("School Type",
            ["All"] + sorted(df_full["school_type"].dropna().unique().tolist()))

        income_order = ["All","Low","Lower-Middle","Middle","Upper-Middle","High"]
        income_opts  = [i for i in income_order if i == "All" or i in df_full["family_income"].dropna().unique()]
        sel_income = st.selectbox("Family Income", income_opts)

        score_range = st.slider("Exam Score Range", 0, 100, (0, 100))

        st.markdown("---")
        st.caption(f"Dataset: **{len(df_full):,} students** · 44 features")

    # Apply filters
    df = df_full.copy()
    if sel_gender != "All":
        df = df[df["gender"] == sel_gender]
    if sel_edu != "All":
        df = df[df["education_level"] == sel_edu]
    if sel_school != "All":
        df = df[df["school_type"] == sel_school]
    if sel_income != "All":
        df = df[df["family_income"] == sel_income]
    df = df[df["exam_score"].between(score_range[0], score_range[1], inclusive="both")]

    # Route to page
    routes = {
        "📊 Overview":            page_overview,
        "👥 Demographics":        page_demographics,
        "📚 Study Habits":        page_study_habits,
        "📈 Performance Analysis": page_performance_analysis,
        "🤖 ML Prediction":       page_ml_prediction,
    }
    routes[page](df, df_full)


if __name__ == "__main__":
    main()
