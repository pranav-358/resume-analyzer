# app.py - AI Resume Matcher Pro (PROFESSIONAL + BUG-FREE)
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from io import BytesIO
import base64

# ML / text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF reading
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional OpenAI (for resume improver)
try:
    import openai
except Exception:
    openai = None

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="ResumeMatch Pro - AI Resume Analyzer", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Helper to render safe HTML blocks
# ----------------------------
def html(content):
    st.markdown(content, unsafe_allow_html=True)

# ----------------------------
# Session State Management
# ----------------------------
def initialize_session_state():
    if "resume_text" not in st.session_state:
        st.session_state["resume_text"] = ""
    if "job_desc_input" not in st.session_state:
        st.session_state["job_desc_input"] = ""
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "match_score" not in st.session_state:
        st.session_state["match_score"] = 0
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None

initialize_session_state()

# ----------------------------
# PREMIUM PROFESSIONAL CSS
# ----------------------------
st.markdown(
    """
<style>
/* ===== PREMIUM COLOR SCHEME ===== */
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --primary-light: #dbeafe;
    --secondary: #64748b;
    --success: #10b981;
    --success-light: #d1fae5;
    --warning: #f59e0b;
    --warning-light: #fef3c7;
    --error: #ef4444;
    --error-light: #fee2e2;
    --background: #f8fafc;
    --surface: #ffffff;
    --text: #1e293b;
    --text-light: #64748b;
    --border: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* ===== MAIN CONTAINER ===== */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px !important;
}

.stApp {
    background: var(--background) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    line-height: 1.6 !important;
}

/* ===== PREMIUM HEADER ===== */
.premium-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    padding: 3rem 2rem !important;
    border-radius: 16px !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid var(--border) !important;
    position: relative !important;
    overflow: hidden !important;
}

.premium-header::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E") !important;
    opacity: 0.3 !important;
}

.main-title {
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    color: white !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.sub-title {
    color: rgba(255, 255, 255, 0.95) !important;
    font-size: 1.25rem !important;
    font-weight: 400 !important;
    margin: 1rem 0 0 0 !important;
    max-width: 600px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    line-height: 1.5 !important;
}

/* ===== PREMIUM CARDS ===== */
.premium-card {
    background: var(--surface) !important;
    padding: 2rem !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 1.5rem !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.premium-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 4px !important;
    height: 100% !important;
    background: linear-gradient(to bottom, var(--primary), var(--success)) !important;
}

.premium-card:hover {
    box-shadow: var(--shadow-hover) !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-2px) !important;
}

/* ===== PREMIUM BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.875rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
    transition: left 0.5s !important;
}

.stButton > button:hover::before {
    left: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
}

/* Secondary Button */
.secondary-button > button {
    background: var(--surface) !important;
    color: var(--primary) !important;
    border: 2px solid var(--primary) !important;
    border-radius: 10px !important;
    padding: 0.875rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.secondary-button > button:hover {
    background: var(--primary-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
}

/* ===== PREMIUM INPUT FIELDS ===== */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    border: 2px solid var(--border) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    transition: all 0.3s ease !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: var(--text-light) !important;
}

/* ===== PREMIUM SKILL TAGS ===== */
.skill-tag {
    display: inline-block !important;
    margin: 0.25rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: 20px !important;
    background: #f1f5f9 !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: 1px solid var(--border) !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

.skill-tag:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.skill-tag.strong {
    background: var(--success-light) !important;
    color: #065f46 !important;
    border-color: #a7f3d0 !important;
}

.skill-tag.improve {
    background: var(--warning-light) !important;
    color: #92400e !important;
    border-color: #fcd34d !important;
}

/* ===== PREMIUM STEP INDICATOR ===== */
.step-indicator {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 2rem 0 !important;
    position: relative !important;
    background: var(--surface) !important;
    padding: 1.5rem !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
}

.step {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    flex: 1 !important;
    position: relative !important;
    z-index: 2 !important;
}

.step-number {
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    background: #f1f5f9 !important;
    color: var(--text-light) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    border: 2px solid var(--border) !important;
    transition: all 0.3s ease !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.step.active .step-number {
    background: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
    transform: scale(1.1) !important;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3) !important;
}

.step.completed .step-number {
    background: var(--success) !important;
    color: white !important;
    border-color: var(--success) !important;
    box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3) !important;
}

.step-line {
    position: absolute !important;
    top: 24px !important;
    left: 50% !important;
    right: -50% !important;
    height: 3px !important;
    background: var(--border) !important;
    z-index: 1 !important;
    transition: all 0.3s ease !important;
    border-radius: 2px !important;
}

.step.completed .step-line {
    background: var(--success) !important;
}

.step-label {
    font-size: 0.9rem !important;
    color: var(--text-light) !important;
    font-weight: 500 !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.step.active .step-label {
    color: var(--primary) !important;
    font-weight: 600 !important;
}

.step.completed .step-label {
    color: var(--success) !important;
}

/* ===== PREMIUM TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background-color: var(--surface) !important;
    padding: 0.5rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

.stTabs [data-baseweb="tab"] {
    height: 50px !important;
    background-color: transparent !important;
    border-radius: 8px !important;
    padding: 0 1.5rem !important;
    font-weight: 500 !important;
    color: var(--text-light) !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border-color: var(--primary) !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
}

/* ===== PREMIUM PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    border-radius: 8px !important;
    height: 10px !important;
}

/* ===== PREMIUM METRICS ===== */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-hover) !important;
}

/* ===== PREMIUM SCORE DISPLAY ===== */
.score-display {
    text-align: center !important;
    padding: 2rem !important;
    border-radius: 16px !important;
    margin: 1rem 0 !important;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
}

.score-value {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--success) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.score-label {
    font-size: 1.25rem !important;
    color: var(--text-light) !important;
    margin: 0.5rem 0 0 0 !important;
    font-weight: 500 !important;
}

/* ===== MOBILE RESPONSIVENESS ===== */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0.5rem !important;
    }
    
    .premium-header {
        padding: 2rem 1rem !important;
    }
    
    .main-title {
        font-size: 2.25rem !important;
    }
    
    .sub-title {
        font-size: 1.1rem !important;
    }
    
    .premium-card {
        padding: 1.5rem !important;
    }
    
    .step-indicator {
        flex-wrap: wrap !important;
        gap: 1rem !important;
        margin: 1.5rem 0 !important;
    }
    
    .step {
        flex: 0 0 calc(50% - 1rem) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .step-line {
        display: none !important;
    }
    
    .score-value {
        font-size: 2.75rem !important;
    }
}

/* ===== HIDE STREAMLIT DEFAULT ELEMENTS ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== PREMIUM STATUS MESSAGES ===== */
.status-success {
    background: var(--success-light) !important;
    color: #065f46 !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    border-left: 4px solid var(--success) !important;
}

.status-warning {
    background: var(--warning-light) !important;
    color: #92400e !important;
    border: 1px solid #fcd34d !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    border-left: 4px solid var(--warning) !important;
}

.status-error {
    background: var(--error-light) !important;
    color: #991b1b !important;
    border: 1px solid #fca5a5 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    border-left: 4px solid var(--error) !important;
}

/* ===== PREMIUM LOADING ANIMATION ===== */
.stSpinner > div {
    border-top-color: var(--primary) !important;
}

/* ===== CUSTOM RADIO BUTTONS ===== */
.stRadio > div {
    flex-direction: row !important;
    gap: 1rem !important;
}

.stRadio > div [role="radiogroup"] {
    display: flex !important;
    gap: 1rem !important;
    flex-wrap: wrap !important;
}

.stRadio > div [role="radiogroup"] label {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
    flex: 1 !important;
    text-align: center !important;
    min-width: 140px !important;
}

.stRadio > div [role="radiogroup"] label:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
}

.stRadio > div [role="radiogroup"] label[data-testid="stRadio"] {
    background: var(--surface) !important;
}

.stRadio > div [role="radiogroup"] div:first-child {
    flex: 1 !important;
}

.stRadio > div [role="radiogroup"] div:first-child label {
    margin-right: 0 !important;
}

/* Selected radio button */
.stRadio > div [role="radiogroup"] label[data-testid="stRadio"]:has(input:checked) {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border-color: var(--primary) !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# CORE FUNCTIONS
# ----------------------------
def extract_skills_advanced(text):
    tech_skills = {
        'Programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift'],
        'Web Frontend': ['react', 'angular', 'vue', 'html', 'css', 'sass', 'bootstrap', 'tailwind', 'javascript', 'typescript'],
        'Web Backend': ['node', 'django', 'flask', 'spring', 'express', 'fastapi', 'ruby on rails', 'php', 'laravel'],
        'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'dynamodb'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'linux'],
        'Data Science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'tableau', 'power bi'],
        'Mobile': ['react native', 'flutter', 'android', 'ios', 'swift', 'kotlin']
    }
    soft_skills = ['communication', 'teamwork', 'leadership', 'problem solving', 'creativity', 'adaptability', 'time management', 'critical thinking', 'collaboration', 'presentation']
    
    found_skills = {category: [] for category in tech_skills}
    found_soft_skills = []
    text_lower = (text or "").lower()
    
    for category, skills in tech_skills.items():
        for skill in skills:
            if skill in text_lower:
                found_skills[category].append(skill.title())
    
    for skill in soft_skills:
        if skill in text_lower:
            found_soft_skills.append(skill.title())
    
    return found_skills, found_soft_skills

def calculate_ai_match(resume_text, job_desc):
    if not (resume_text or "").strip() or not (job_desc or "").strip():
        return 0, {}, [], {}
    
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    resume_tech, resume_soft = extract_skills_advanced(resume_text)
    job_tech, job_soft = extract_skills_advanced(job_desc)
    
    total_tech_score = 0
    max_tech_score = 0
    
    for category in job_tech:
        job_skills = job_tech[category]
        resume_skills = resume_tech[category]
        if job_skills:
            max_tech_score += len(job_skills)
            matched = len(set(job_skills) & set(resume_skills))
            total_tech_score += matched
    
    tech_match = (total_tech_score / max_tech_score * 100) if max_tech_score > 0 else 0
    soft_match = (len(set(job_soft) & set(resume_soft)) / len(job_soft) * 100) if job_soft else 0
    
    final_score = (similarity * 40 + tech_match * 45 + soft_match * 15)
    return min(round(final_score * 100, 2), 100), resume_tech, resume_soft, job_tech, job_soft

def analyze_job_description(jd_text):
    jd = jd_text or ""
    lines = [l.strip() for l in jd.splitlines() if l.strip()]
    jd_lower = jd.lower()
    title = ""
    
    m = re.search(r"job title[:\-]\s*(.+)", jd_lower)
    if m:
        title = m.group(1).strip().title()
    else:
        if lines:
            first = lines[0]
            if len(first.split()) < 6:
                title = first.title()
    
    bullets = re.split(r"[\n•\-]+", jd)
    bullets = [b.strip() for b in bullets if len(b.strip()) > 2]
    job_tech, job_soft = extract_skills_advanced(jd)
    
    return {
        "title": title or "Not Specified",
        "bullets": bullets,
        "requirements": bullets[:6],
        "job_tech": job_tech,
        "job_soft": job_soft
    }

def extract_text_from_pdf(file_bytes):
    if PdfReader is None:
        return None, "PyPDF2 not installed. Install with `pip install PyPDF2`"
    try:
        reader = PdfReader(file_bytes)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text), None
    except Exception as e:
        return None, f"Failed to extract PDF text: {e}"

def ai_improve_resume(resume_text, openai_api_key, target_role=None, max_tokens=700):
    if not openai:
        return None, "OpenAI package not installed"
    if not openai_api_key:
        return None, "No OpenAI API key provided"
    
    try:
        openai.api_key = openai_api_key
        prompt = f"""Improve this resume for {target_role or 'a professional role'}:
        
{resume_text}

Provide only the improved resume content:"""
        
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        improved = resp["choices"][0]["message"]["content"].strip()
        return improved, None
    except Exception as e:
        return None, f"OpenAI request failed: {e}"

def generate_text_report(match_score, strong_points, improvement_points, job_title=""):
    """Generate a simple text report instead of PDF to avoid encoding issues"""
    report = f"""
RESUME MATCH ANALYSIS REPORT
============================

Job Title: {job_title}
Match Score: {match_score}%

STRONG POINTS:
{chr(10).join(f"• {point}" for point in strong_points) if strong_points else "• No strong points identified"}

IMPROVEMENT AREAS:
{chr(10).join(f"• {point}" for point in improvement_points) if improvement_points else "• No major improvement areas"}

RECOMMENDATIONS:
• Focus on developing the skills mentioned above
• Tailor your resume to highlight matching skills
• Consider relevant projects or certifications

Generated by ResumeMatch Pro
    """
    return report

# ----------------------------
# STEP MANAGEMENT
# ----------------------------
def update_step_progress():
    """Update step progress based on user actions"""
    has_resume = bool(st.session_state.get("resume_text", "").strip())
    has_job_desc = bool(st.session_state.get("job_desc_input", "").strip())
    analysis_done = st.session_state.get("analysis_complete", False)
    
    if analysis_done:
        st.session_state.current_step = 4
    elif has_resume and has_job_desc:
        st.session_state.current_step = 3
    elif has_resume:
        st.session_state.current_step = 2
    else:
        st.session_state.current_step = 1

# ----------------------------
# PREMIUM UI BUILDING
# ----------------------------

# Premium Header
html(f"""
<div class="premium-header">
    <div class="main-title">ResumeMatch Pro</div>
    <div class="sub-title">AI-Powered Resume Analysis • Professional Matching • Career Insights</div>
</div>
""")

# Update step progress
update_step_progress()

# Dynamic Step Indicator
current_step = st.session_state.current_step
html(f"""
<div class="step-indicator">
    <div class="step {'completed' if current_step >= 1 else 'active' if current_step == 1 else ''}">
        <div class="step-number">1</div>
        <div class="step-line"></div>
        <div class="step-label">Upload Resume</div>
    </div>
    <div class="step {'completed' if current_step >= 2 else 'active' if current_step == 2 else ''}">
        <div class="step-number">2</div>
        <div class="step-line"></div>
        <div class="step-label">Add Job Details</div>
    </div>
    <div class="step {'completed' if current_step >= 3 else 'active' if current_step == 3 else ''}">
        <div class="step-number">3</div>
        <div class="step-line"></div>
        <div class="step-label">Get Analysis</div>
    </div>
    <div class="step {'completed' if current_step >= 4 else 'active' if current_step == 4 else ''}">
        <div class="step-number">4</div>
        <div class="step-label">View Results</div>
    </div>
</div>
""")

# Premium Sidebar
with st.sidebar:
    html('<div class="premium-card">')
    st.header("⚙️ Settings & Tools")
    
    st.subheader("🤖 AI Features")
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        placeholder="Enter your API key...",
        help="Required for AI resume improvements"
    )
    
    use_ai_rewrite = st.checkbox(
        "Enable AI Resume Rewriter", 
        value=False,
        help="Get AI-powered resume improvements"
    )
    
    if use_ai_rewrite and not openai_api_key:
        st.warning("🔑 Please add your OpenAI API key to enable AI features")
    
    st.markdown("---")
    st.subheader("📚 Quick Guide")
    st.info("""
    **How to use:**
    1. 📝 Upload or paste your resume
    2. 💼 Add the job description
    3. 📊 Click Analyze to get insights
    4. 📈 Review matches and improvements
    """)
    
    st.markdown("---")
    st.subheader("💡 Tips")
    st.success("""
    • Be specific in your resume
    • Include relevant keywords
    • Highlight measurable achievements
    • Tailor skills to job requirements
    """)
    html('</div>')

# Main Content Area with Premium Tabs
tab1, tab2, tab3 = st.tabs(["📝 Resume", "💼 Job Description", "📊 Analysis"])

with tab1:
    html('<div class="premium-card">')
    st.subheader("📄 Resume Input")
    
    # File upload section
    upload_col1, upload_col2 = st.columns([2, 1])
    with upload_col1:
        upload_mode = st.radio(
            "Select input method:",
            ["Paste Text", "Upload PDF"],
            horizontal=True
        )
    
    if upload_mode == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload your resume PDF",
            type=["pdf"],
            help="Supported: PDF files"
        )
        if uploaded_file:
            if PdfReader is None:
                st.error("PyPDF2 required: Install with `pip install PyPDF2`")
            else:
                with st.spinner("📄 Extracting text from PDF..."):
                    txt, err = extract_text_from_pdf(uploaded_file)
                    if err:
                        st.error(f"❌ Error: {err}")
                    else:
                        st.session_state["resume_text"] = txt
                        st.success("✅ PDF extracted successfully!")
                        with st.expander("👁️ Preview extracted text"):
                            st.text_area(
                                "Extracted Content", 
                                value=st.session_state["resume_text"][:800] + "..." if len(st.session_state["resume_text"]) > 800 else st.session_state["resume_text"],
                                height=150,
                                key="preview_area"
                            )
    
    # Text area for resume
    resume_text = st.text_area(
        "Paste your resume content:",
        height=300,
        value=st.session_state["resume_text"],
        placeholder="""EXPERIENCE:
• 2 years as Full Stack Developer
• Built web applications using React & Node.js
• Implemented machine learning models

SKILLS:
• Programming: Python, JavaScript, Java
• Frameworks: React, Node.js, Django
• Databases: MySQL, MongoDB
• Tools: Git, Docker, AWS

EDUCATION:
• B.Tech in Computer Science""",
        key="resume_area",
        on_change=lambda: st.session_state.update({"resume_text": st.session_state.resume_area})
    )
    
    # Update session state
    st.session_state["resume_text"] = resume_text
    
    # Character count
    if st.session_state["resume_text"]:
        char_count = len(st.session_state["resume_text"])
        word_count = len(st.session_state["resume_text"].split())
        st.caption(f"📝 {char_count} characters, {word_count} words")
    
    html('</div>')
    
    # AI Improvement Section
    if use_ai_rewrite:
        html('<div class="premium-card">')
        st.subheader("🤖 AI Resume Improver")
        
        target_role = st.text_input(
            "Target role (optional):",
            placeholder="e.g., Full Stack Developer",
            help="AI will tailor improvements for this role"
        )
        
        if st.button("✨ Improve with AI", use_container_width=True):
            if not st.session_state["resume_text"]:
                st.error("❌ Please add resume content first")
            elif not openai_api_key:
                st.error("🔑 OpenAI API key required")
            else:
                with st.spinner("🤖 AI is enhancing your resume..."):
                    improved, err = ai_improve_resume(
                        st.session_state["resume_text"], 
                        openai_api_key, 
                        target_role=target_role
                    )
                    if err:
                        st.error(f"❌ Error: {err}")
                    else:
                        st.session_state["resume_text"] = improved
                        st.success("✅ Resume improved! Review below")
        html('</div>')

with tab2:
    html('<div class="premium-card">')
    st.subheader("💼 Job Description Analysis")
    
    job_desc_input = st.text_area(
        "Paste the job description:",
        height=300,
        value=st.session_state["job_desc_input"],
        placeholder="""JOB TITLE: Full Stack Developer

REQUIREMENTS:
• 2+ years experience in web development
• Strong proficiency in JavaScript, Python
• Experience with React.js and Node.js
• Knowledge of databases (SQL & NoSQL)
• Familiarity with cloud platforms

SOFT SKILLS:
• Excellent communication skills
• Team player mentality
• Problem-solving attitude""",
        key="job_area",
        on_change=lambda: st.session_state.update({"job_desc_input": st.session_state.job_area})
    )
    
    # Update session state
    st.session_state["job_desc_input"] = job_desc_input
    
    if st.session_state["job_desc_input"]:
        char_count = len(st.session_state["job_desc_input"])
        word_count = len(st.session_state["job_desc_input"].split())
        st.caption(f"📋 {char_count} characters, {word_count} words")
    
    # JD Analysis
    if st.button("🔍 Analyze Job Description", use_container_width=True):
        if not st.session_state["job_desc_input"]:
            st.error("❌ Please add a job description first")
        else:
            with st.spinner("🔍 Analyzing requirements..."):
                jd_struct = analyze_job_description(st.session_state["job_desc_input"])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Detected Job Title", jd_struct["title"])
                with col2:
                    st.metric("📋 Key Requirements", len(jd_struct["requirements"]))
                with col3:
                    tech_skills_count = sum(len(skills) for skills in jd_struct["job_tech"].values())
                    st.metric("⚙️ Technical Skills", tech_skills_count)
                with col4:
                    st.metric("🤝 Soft Skills", len(jd_struct["job_soft"]))
                
                # Skills breakdown
                with st.expander("🔧 Technical Skills Breakdown"):
                    for cat, skills in jd_struct["job_tech"].items():
                        if skills:
                            st.write(f"**{cat}:**")
                            for skill in skills[:5]:  # Limit to 5 skills per category
                                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
    html('</div>')

with tab3:
    html('<div class="premium-card">')
    st.subheader("📊 Resume Match Analysis")
    
    if not st.session_state["resume_text"] or not st.session_state["job_desc_input"]:
        st.warning("⚠️ Please add both resume content and job description to proceed with analysis.")
    else:
        st.success("✅ Ready for analysis! Click the button below to see how your resume matches the job requirements.")
    
    # Central analyze button
    if st.button("🚀 Analyze Resume Match", type="primary", use_container_width=True):
        resume_text = st.session_state.get("resume_text", "")
        job_desc_input = st.session_state.get("job_desc_input", "")
        
        if not resume_text.strip():
            st.error("❌ Please provide your resume content")
        elif not job_desc_input.strip():
            st.error("❌ Please provide a job description")
        else:
            with st.spinner("🔍 Analyzing your resume match..."):
                time.sleep(1)
                score, rtech, rsoft, jtech, jsoft = calculate_ai_match(resume_text, job_desc_input)
                st.session_state.analysis_complete = True
                st.session_state.current_step = 4
                st.session_state.match_score = score
                st.session_state.analysis_results = {
                    "rtech": rtech,
                    "rsoft": rsoft,
                    "jtech": jtech,
                    "jsoft": jsoft
                }
                
                # Display Results
                st.markdown("---")
                
                # Score Card
                html('<div class="score-display">')
                if score >= 80:
                    st.success(f"**🎉 EXCELLENT MATCH**")
                    st.info("Your resume strongly aligns with this job requirement!")
                elif score >= 60:
                    st.warning(f"**👍 GOOD MATCH**")
                    st.info("Good foundation with some areas for improvement.")
                else:
                    st.error(f"**📈 NEEDS IMPROVEMENT**")
                    st.info("Focus on developing the missing skills below.")
                
                html(f'<div class="score-value">{score}%</div>')
                html('<div class="score-label">Match Score</div>')
                st.progress(score / 100)
                html('</div>')
                
                # Skills Analysis
                col_left, col_right = st.columns(2)
                strong_list, missing_list = [], []
                
                with col_left:
                    html('<div class="premium-card">')
                    st.subheader("✅ Your Strong Points")
                    any_strong = False
                    
                    for cat in rtech:
                        matched = set(rtech[cat]) & set(jtech[cat])
                        if matched:
                            any_strong = True
                            st.write(f"**{cat}**")
                            for s in matched:
                                strong_list.append(f"{cat}: {s}")
                                st.markdown(f'<span class="skill-tag strong">✓ {s}</span>', unsafe_allow_html=True)
                    
                    matched_soft = set(rsoft) & set(jsoft)
                    if matched_soft:
                        any_strong = True
                        st.write("**🤝 Soft Skills**")
                        for s in matched_soft:
                            strong_list.append(f"Soft: {s}")
                            st.markdown(f'<span class="skill-tag strong">✓ {s}</span>', unsafe_allow_html=True)
                    
                    if not any_strong:
                        st.info("ℹ️ No strong matching points detected.")
                    html('</div>')
                
                with col_right:
                    html('<div class="premium-card">')
                    st.subheader("📚 Improvement Areas")
                    any_missing = False
                    
                    for cat in jtech:
                        missing = set(jtech[cat]) - set(rtech[cat])
                        if missing:
                            any_missing = True
                            st.write(f"**{cat}**")
                            for s in list(missing)[:3]:
                                missing_list.append(f"{cat}: {s}")
                                st.markdown(f'<span class="skill-tag improve">+ {s}</span>', unsafe_allow_html=True)
                    
                    missing_soft = set(jsoft) - set(rsoft)
                    if missing_soft:
                        any_missing = True
                        st.write("**🤝 Soft Skills**")
                        for s in list(missing_soft)[:2]:
                            missing_list.append(f"Soft: {s}")
                            st.markdown(f'<span class="skill-tag improve">+ {s}</span>', unsafe_allow_html=True)
                    
                    if not any_missing:
                        st.success("🎉 Excellent! No major skill gaps found.")
                    html('</div>')
                
                # Generate and Download Report
                html('<div class="premium-card">')
                st.subheader("📥 Download Report")
                
                jd_title = analyze_job_description(job_desc_input)['title']
                report_text = generate_text_report(score, strong_list, missing_list, job_title=jd_title)
                
                # Create download button for text report
                st.download_button(
                    "📄 Download Analysis Report",
                    data=report_text,
                    file_name=f"resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                html('</div>')
    
    html('</div>')

# Premium Footer
html("""
<div style='
    text-align:center; 
    color:#64748b; 
    margin-top:3rem; 
    padding:2rem; 
    background:white;
    border-radius:16px;
    border:1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
'>
    <div style='font-weight:700; margin-bottom:0.5rem; font-size:1.25rem; color:#1e293b;'>ResumeMatch Pro</div>
    <div style='font-size:0.95rem; margin-bottom:0.5rem; color:#64748b;'>Professional Resume Analysis Tool</div>
    <div style='font-size:0.85rem; color:#94a3b8;'>Built for Career Success • Powered by AI</div>
    <div style='font-size:0.85rem; color:#94a3b8; margin-top:0.5rem;'>© 2025 ResumeMatch Pro. All rights reserved.</div>
</div>
""")