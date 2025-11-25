# app.py - AI Resume Matcher Pro (ULTRA ATTRACTIVE + MOBILE FRIENDLY)
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from io import BytesIO
from fpdf import FPDF

# ML / text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF reading
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Lottie (optional)
try:
    from streamlit_lottie import st_lottie
    import requests
except Exception:
    st_lottie = None
    requests = None

# Optional OpenAI (for resume improver)
try:
    import openai
except Exception:
    openai = None

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="ResumeMatch AI - Smart Resume Analysis", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Helper to render safe HTML blocks
# ----------------------------
def html(content):
    st.markdown(content, unsafe_allow_html=True)

# ----------------------------
# Utilities
# ----------------------------
def load_lottie_url_safe(url):
    if not requests:
        return None
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None

lottie_tech = load_lottie_url_safe("https://assets1.lottiefiles.com/packages/lf20_ukgq8loj.json")
lottie_success = load_lottie_url_safe("https://assets1.lottiefiles.com/packages/lf20_x17ybolp.json")
lottie_upload = load_lottie_url_safe("https://assets1.lottiefiles.com/packages/lf20_5tkzkflw.json")

# ----------------------------
# ULTRA ATTRACTIVE CSS WITH ANIMATIONS
# ----------------------------
st.markdown(
    """
<style>
/* ===== FIX STREAMLIT DEFAULT SPACING ===== */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px !important;
}

/* ===== MODERN GRADIENT BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background-attachment: fixed !important;
}

/* ===== GLASS MORPHISM HEADER ===== */
.glass-header {
    background: rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    padding: 2.5rem 3rem !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.1) !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
    animation: fadeInUp 0.8s ease-out !important;
}

.main-title {
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: gradientShift 3s ease infinite !important;
    margin: 0 !important;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
}

.sub-title {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.3rem !important;
    font-weight: 400 !important;
    margin: 1rem 0 0 0 !important;
    animation: fadeIn 1s ease-out 0.3s both !important;
}

/* ===== GLASS MORPHISM CARDS ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    padding: 2.5rem !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    margin-bottom: 2rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: slideInUp 0.6s ease-out !important;
}

.glass-card:hover {
    transform: translateY(-8px) scale(1.01) !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
}

/* ===== ANIMATED BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 1rem 2.5rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3) !important;
    width: 100% !important;
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
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4) !important;
    background: linear-gradient(135deg, #FF8E8E 0%, #6BE8E1 100%) !important;
}

/* ===== CLEAR VISIBLE TEXT AREAS ===== */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px !important;
    padding: 1.2rem !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    transition: all 0.3s ease !important;
    color: #2D3748 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

.stTextArea textarea:focus {
    border-color: #4ECDC4 !important;
    box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.2) !important;
    background: white !important;
    transform: scale(1.02) !important;
}

.stTextArea textarea::placeholder {
    color: #718096 !important;
    font-style: italic !important;
}

/* ===== ANIMATED SKILL TAGS ===== */
.skill-tag {
    display: inline-block !important;
    margin: 0.4rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 25px !important;
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: popIn 0.5s ease-out !important;
    cursor: pointer !important;
}

.skill-tag:hover {
    transform: scale(1.1) rotate(2deg) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
}

/* ===== FLOATING STEP INDICATOR ===== */
.step-indicator {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 3rem 0 !important;
    position: relative !important;
    animation: fadeInUp 0.8s ease-out 0.2s both !important;
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
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-weight: 800 !important;
    margin-bottom: 0.8rem !important;
    border: 3px solid rgba(255, 255, 255, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    font-size: 1.2rem !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
}

.step.active .step-number {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%) !important;
    transform: scale(1.2) !important;
    box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    animation: pulse 2s infinite !important;
}

.step-line {
    position: absolute !important;
    top: 30px !important;
    left: 50% !important;
    right: -50% !important;
    height: 4px !important;
    background: rgba(255, 255, 255, 0.2) !important;
    z-index: 1 !important;
    transition: all 0.4s ease !important;
}

.step.active .step-line {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4) !important;
    box-shadow: 0 2px 10px rgba(255, 107, 107, 0.3) !important;
}

.step-label {
    font-size: 1rem !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
    text-align: center !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

.step.active .step-label {
    color: white !important;
    transform: scale(1.1) !important;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
}

/* ===== ANIMATED TAB STYLING ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
    padding: 0.8rem !important;
    border-radius: 20px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.stTabs [data-baseweb="tab"] {
    height: 60px !important;
    background-color: transparent !important;
    border-radius: 15px !important;
    padding: 0 1.5rem !important;
    font-weight: 600 !important;
    color: rgba(255, 255, 255, 0.8) !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%) !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* ===== ANIMATED PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%) !important;
    border-radius: 10px !important;
    animation: progressFill 2s ease-in-out !important;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(50px) scale(0.9);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes popIn {
    0% { transform: scale(0); opacity: 0; }
    70% { transform: scale(1.1); }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(255, 107, 107, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
}

@keyframes progressFill {
    from { width: 0% !important; }
    to { width: 100% !important; }
}

/* ===== MOBILE RESPONSIVENESS ===== */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0.5rem !important;
    }
    
    .glass-header {
        padding: 1.5rem 1rem !important;
        border-radius: 20px !important;
    }
    
    .main-title {
        font-size: 2rem !important;
    }
    
    .sub-title {
        font-size: 1rem !important;
    }
    
    .glass-card {
        padding: 1.5rem !important;
        border-radius: 15px !important;
    }
    
    .step-indicator {
        flex-wrap: wrap !important;
        gap: 1rem !important;
        margin: 2rem 0 !important;
    }
    
    .step {
        flex: 0 0 calc(50% - 1rem) !important;
        margin-bottom: 1rem !important;
    }
    
    .step-number {
        width: 50px !important;
        height: 50px !important;
        font-size: 1rem !important;
    }
    
    .step-line {
        display: none !important;
    }
    
    .stButton > button {
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
    }
    
    .skill-tag {
        font-size: 0.8rem !important;
        padding: 0.5rem 1rem !important;
    }
}

@media (max-width: 480px) {
    .step {
        flex: 0 0 100% !important;
    }
    
    .main-title {
        font-size: 1.6rem !important;
    }
}

/* ===== HIDE STREAMLIT DEFAULT ELEMENTS ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== ANIMATED METRICS ===== */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
    animation: slideInUp 0.6s ease-out !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15) !important;
}

/* ===== FLOATING ACTION BUTTON ===== */
.floating-btn {
    position: fixed !important;
    bottom: 2rem !important;
    right: 2rem !important;
    z-index: 1000 !important;
    animation: bounce 2s infinite !important;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
    40% {transform: translateY(-10px);}
    60% {transform: translateY(-5px);}
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# REST OF YOUR FUNCTIONS (UNCHANGED)
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
        "title": title or "Unknown",
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
        return None, "openai package not installed. Install with `pip install openai`"
    if not openai_api_key:
        return None, "No OpenAI API key provided. Set it in sidebar or set OPENAI_API_KEY env var."
    try:
        openai.api_key = openai_api_key
        prompt = f"""You are a professional resume editor. Improve the following resume content to be concise, professional, action-oriented, and tailored for {target_role or 'the target job'}.
Keep bullet points, improve verbs, fix grammar, and suggest a short 2-line summary at top.

Resume:
{resume_text}

Provide output only in plain text with improved resume content."""
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

def generate_report_pdf(match_score, resume_text, job_desc, strong_points, improvement_points, job_title=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "AI Resume Matcher Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, f"Job Title: {job_title}", ln=True)
    pdf.cell(0, 6, f"Match Score: {match_score}%", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Strong Points:", ln=True)
    pdf.set_font("Helvetica", size=11)
    if strong_points:
        for s in strong_points:
            pdf.multi_cell(0, 6, f"- {s}")
    else:
        pdf.multi_cell(0,6, "- None detected")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Improvement Areas:", ln=True)
    pdf.set_font("Helvetica", size=11)
    if improvement_points:
        for s in improvement_points:
            pdf.multi_cell(0, 6, f"- {s}")
    else:
        pdf.multi_cell(0,6, "- None detected")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0,6, "Extracts from Resume:", ln=True)
    pdf.set_font("Helvetica", size=10)
    sample_resume = (resume_text or "")[:1500]
    pdf.multi_cell(0,5, sample_resume)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0,6,"Job Description Extract:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0,5, (job_desc or "")[:1500])
    out = pdf.output(dest="S").encode("latin-1")
    return out

# ----------------------------
# ULTRA ATTRACTIVE UI BUILDING
# ----------------------------

# ANIMATED GLASS HEADER
html(f"""
<div class="glass-header">
    <div class="main-title">🚀 ResumeMatch AI</div>
    <div class="sub-title">Smart Resume Analysis • AI-Powered Matching • Professional Reports</div>
</div>
""")

# FLOATING STEP INDICATOR
html("""
<div class="step-indicator">
    <div class="step active">
        <div class="step-number">1</div>
        <div class="step-line"></div>
        <div class="step-label">Upload Resume</div>
    </div>
    <div class="step">
        <div class="step-number">2</div>
        <div class="step-line"></div>
        <div class="step-label">Add Job Details</div>
    </div>
    <div class="step">
        <div class="step-number">3</div>
        <div class="step-line"></div>
        <div class="step-label">Get Analysis</div>
    </div>
    <div class="step">
        <div class="step-number">4</div>
        <div class="step-line"></div>
        <div class="step-label">Download Report</div>
    </div>
</div>
""")

# Enhanced Sidebar with Glass Effect
with st.sidebar:
    html('<div class="glass-card">')
    st.header("⚙️ Settings & Tools")
    
    st.subheader("AI Features")
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        placeholder="sk-...",
        help="Optional: For AI resume improvements"
    )
    
    use_ai_rewrite = st.checkbox(
        "Enable AI Resume Rewriter", 
        value=False,
        help="Get AI-powered resume improvements"
    )
    
    if use_ai_rewrite and not openai_api_key:
        st.warning("🔑 Please add your OpenAI API key to enable AI features")
    
    st.markdown("---")
    st.subheader("Quick Tips")
    st.info("""
    💡 **Pro Tips:**
    - Upload PDF resumes for quick analysis
    - Use AI rewrite for professional improvements  
    - Download PDF reports for interviews
    - Analyze job descriptions before applying
    """)
    html('</div>')

# Initialize session state
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "job_desc_input" not in st.session_state:
    st.session_state["job_desc_input"] = ""
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1

# Main Content Area with Animated Tabs
tab1, tab2, tab3 = st.tabs(["📝 Resume Input", "💼 Job Analysis", "📊 Match Results"])

with tab1:
    html('<div class="glass-card">')
    
    # Header with animation
    col_header = st.columns([3, 1])
    with col_header[0]:
        st.subheader("📄 Your Resume Content")
    with col_header[1]:
        if lottie_upload:
            st_lottie(lottie_upload, height=80, key="upload_anim")
    
    # File upload section with better styling
    st.markdown("### 📥 Choose Input Method")
    upload_mode = st.radio(
        "",
        ["📋 Paste Text", "📁 Upload PDF"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if upload_mode == "📁 Upload PDF":
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload your resume PDF",
            type=["pdf"],
            help="Supported: PDF files up to 10MB"
        )
        if uploaded_file:
            if PdfReader is None:
                st.error("📚 PyPDF2 required: Run `pip install PyPDF2`")
            else:
                with st.spinner("📖 Reading your PDF..."):
                    txt, err = extract_text_from_pdf(uploaded_file)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["resume_text"] = txt
                        st.success("✅ PDF extracted successfully!")
                        with st.expander("👀 Preview extracted text", expanded=True):
                            st.text_area(
                                "Extracted Content", 
                                value=st.session_state["resume_text"][:1000] + "..." if len(st.session_state["resume_text"]) > 1000 else st.session_state["resume_text"],
                                height=200,
                                key="preview_area"
                            )
    
    # Text area for resume - NOW PERFECTLY VISIBLE
    st.markdown("### ✍️ Your Resume Content")
    st.session_state["resume_text"] = st.text_area(
        "Paste or edit your resume content below:",
        height=300,
        value=st.session_state["resume_text"],
        placeholder="""EXPERIENCE:
• 2 years as Full Stack Developer at Tech Corp
• Built web applications using React & Node.js
• Implemented machine learning models with Python

SKILLS:
• Programming: Python, JavaScript, Java
• Frameworks: React, Node.js, Django
• Databases: MySQL, MongoDB, PostgreSQL
• Tools: Git, Docker, AWS

EDUCATION:
• B.Tech in Computer Science - 8.5 CGPA""",
        key="resume_area",
        label_visibility="collapsed"
    )
    
    # Character count with animation
    if st.session_state["resume_text"]:
        char_count = len(st.session_state["resume_text"])
        word_count = len(st.session_state["resume_text"].split())
        col_stats = st.columns(3)
        with col_stats[0]:
            st.metric("📝 Characters", char_count)
        with col_stats[1]:
            st.metric("📊 Words", word_count)
        with col_stats[2]:
            st.metric("📄 Pages", f"{(char_count / 1500):.1f}")
    
    html('</div>')
    
    # AI Improvement Section
    if use_ai_rewrite:
        html('<div class="glass-card">')
        st.subheader("🤖 AI Resume Improver")
        
        target_role = st.text_input(
            "🎯 Target role (optional):",
            placeholder="e.g., Full Stack Developer, Data Scientist",
            help="AI will tailor improvements for this specific role"
        )
        
        if st.button("✨ Improve with AI", use_container_width=True):
            if not st.session_state["resume_text"]:
                st.error("Please add resume content first")
            elif not openai_api_key:
                st.error("OpenAI API key required")
            else:
                with st.spinner("🪄 AI is enhancing your resume..."):
                    improved, err = ai_improve_resume(
                        st.session_state["resume_text"], 
                        openai_api_key, 
                        target_role=target_role
                    )
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["resume_text"] = improved
                        st.success("✅ Resume improved! Review and edit below")
        html('</div>')

with tab2:
    html('<div class="glass-card">')
    st.subheader("💼 Job Description Analysis")
    
    st.session_state["job_desc_input"] = st.text_area(
        "📋 Paste the job description:",
        height=300,
        value=st.session_state["job_desc_input"],
        placeholder="""JOB TITLE: Full Stack Developer

REQUIREMENTS:
• 2+ years experience in web development
• Strong proficiency in JavaScript, Python
• Experience with React.js and Node.js
• Knowledge of databases (SQL & NoSQL)
• Familiarity with cloud platforms (AWS)
• Understanding of CI/CD pipelines

NICE TO HAVE:
• Machine learning experience
• Mobile development (React Native)
• DevOps practices

SOFT SKILLS:
• Excellent communication skills
• Team player mentality
• Problem-solving attitude""",
        key="job_area"
    )
    
    if st.session_state["job_desc_input"]:
        char_count = len(st.session_state["job_desc_input"])
        word_count = len(st.session_state["job_desc_input"].split())
        st.caption(f"📊 {char_count} characters, {word_count} words")
    
    # JD Analysis
    if st.button("🔍 Analyze Job Description", use_container_width=True):
        if not st.session_state["job_desc_input"]:
            st.error("Please add a job description first")
        else:
            with st.spinner("Analyzing job requirements..."):
                jd_struct = analyze_job_description(st.session_state["job_desc_input"])
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("📝 Detected Job Title", jd_struct["title"])
                    st.metric("📋 Key Requirements", len(jd_struct["requirements"]))
                
                with col_b:
                    tech_skills_count = sum(len(skills) for skills in jd_struct["job_tech"].values())
                    st.metric("🛠️ Technical Skills", tech_skills_count)
                    st.metric("💬 Soft Skills", len(jd_struct["job_soft"]))
                
                # Skills breakdown
                with st.expander("🔧 Technical Skills Breakdown", expanded=True):
                    for cat, skills in jd_struct["job_tech"].items():
                        if skills:
                            st.write(f"**{cat}:**")
                            cols = st.columns(3)
                            for i, skill in enumerate(skills):
                                cols[i % 3].markdown(f"<span class='skill-tag'>{skill}</span>", unsafe_allow_html=True)
                
                if jd_struct["job_soft"]:
                    with st.expander("💬 Soft Skills Required", expanded=True):
                        cols = st.columns(3)
                        for i, skill in enumerate(jd_struct["job_soft"]):
                            cols[i % 3].markdown(f"<span class='skill-tag'>{skill}</span>", unsafe_allow_html=True)
    
    html('</div>')

with tab3:
    html('<div class="glass-card">')
    st.subheader("🚀 Ready to Analyze!")
    
    st.info("""
    **Next Steps:**
    1. ✅ Add your resume in the first tab
    2. ✅ Add job description in the second tab  
    3. 🎯 Click the button below to get your match analysis
    4. 📥 Download your personalized report
    """)
    
    # Central analyze button with animation
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        analyze_btn = st.button("🎯 Analyze Resume Match", use_container_width=True, type="primary")
    
    if analyze_btn:
        resume_text = st.session_state.get("resume_text", "")
        job_desc_input = st.session_state.get("job_desc_input", "")
        
        if not resume_text.strip():
            st.error("❌ Please provide your resume content")
        elif not job_desc_input.strip():
            st.error("❌ Please provide a job description")
        else:
            with st.spinner("🔍 Analyzing your match..."):
                time.sleep(1)
                score, rtech, rsoft, jtech, jsoft = calculate_ai_match(resume_text, job_desc_input)
                
                # Display Results
                st.markdown("---")
                
                # Score Card
                html('<div class="glass-card">')
                col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
                
                with col_score2:
                    if score >= 80:
                        st.success(f"🎉 EXCELLENT MATCH - {score}%")
                        if st_lottie and lottie_success:
                            try:
                                st_lottie(lottie_success, height=120)
                            except:
                                pass
                    elif score >= 60:
                        st.warning(f"👍 GOOD MATCH - {score}%")
                    else:
                        st.error(f"💡 NEEDS IMPROVEMENT - {score}%")
                    
                    st.progress(score / 100)
                    
                    # Match interpretation
                    if score >= 80:
                        st.info("**🎯 Your resume strongly aligns with this job!**")
                    elif score >= 60:
                        st.info("**📈 Good foundation - consider some improvements**")
                    else:
                        st.info("**🛠️ Focus on developing missing skills**")
                
                html('</div>')
                
                # Skills Analysis
                col_skills1, col_skills2 = st.columns(2)
                strong_list, missing_list = [], []
                
                with col_skills1:
                    html('<div class="glass-card">')
                    st.subheader("✅ Your Strong Points")
                    any_strong = False
                    for cat in rtech:
                        matched = set(rtech[cat]) & set(jtech[cat])
                        if matched:
                            any_strong = True
                            st.write(f"**{cat}**")
                            for s in matched:
                                strong_list.append(f"{cat}: {s}")
                                st.markdown(f"<span class='skill-tag'>🎯 {s}</span>", unsafe_allow_html=True)
                    matched_soft = set(rsoft) & set(jsoft)
                    if matched_soft:
                        any_strong = True
                        st.write("**Soft Skills**")
                        for s in matched_soft:
                            strong_list.append(f"Soft: {s}")
                            st.markdown(f"<span class='skill-tag'>💬 {s}</span>", unsafe_allow_html=True)
                    if not any_strong:
                        st.info("No strong matching points detected.")
                    html('</div>')
                
                with col_skills2:
                    html('<div class="glass-card">')
                    st.subheader("📚 Improvement Areas")
                    any_missing = False
                    for cat in jtech:
                        missing = set(jtech[cat]) - set(rtech[cat])
                        if missing:
                            any_missing = True
                            st.write(f"**{cat}**")
                            for s in list(missing)[:4]:
                                missing_list.append(f"{cat}: {s}")
                                st.markdown(f"<span class='skill-tag' style='background: linear-gradient(135deg, #ef4444, #dc2626);'>📘 {s}</span>", unsafe_allow_html=True)
                    missing_soft = set(jsoft) - set(rsoft)
                    if missing_soft:
                        any_missing = True
                        st.write("**Soft Skills**")
                        for s in list(missing_soft)[:3]:
                            missing_list.append(f"Soft: {s}")
                            st.markdown(f"<span class='skill-tag' style='background: linear-gradient(135deg, #ef4444, #dc2626);'>💬 {s}</span>", unsafe_allow_html=True)
                    if not any_missing:
                        st.success("🎉 No major skill gaps found!")
                    html('</div>')
                
                # Visualization
                try:
                    import plotly.graph_objects as go
                    cats = list(rtech.keys())
                    resume_counts = [len(rtech[c]) for c in cats]
                    job_counts = [len(jtech[c]) for c in cats]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Your Skills", 
                        x=cats, 
                        y=resume_counts,
                        marker_color='#4ECDC4'
                    ))
                    fig.add_trace(go.Bar(
                        name="Required Skills", 
                        x=cats, 
                        y=job_counts,
                        marker_color='#FF6B6B'
                    ))
                    fig.update_layout(
                        title="Skill Distribution Comparison",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
                
                # Download Report
                html('<div class="glass-card">')
                st.subheader("📥 Download Your Report")
                
                jd_title = analyze_job_description(job_desc_input)['title'] if job_desc_input else ""
                report_bytes = generate_report_pdf(
                    score, resume_text, job_desc_input, 
                    strong_list, missing_list, job_title=jd_title
                )
                
                st.download_button(
                    "💾 Download Comprehensive Report (PDF)",
                    data=report_bytes,
                    file_name=f"resume_match_report_{jd_title.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                html('</div>')
    
    html('</div>')

# Animated Footer
html("""
<div style='
    text-align:center; 
    color:rgba(255,255,255,0.9); 
    margin-top:3rem; 
    padding:2.5rem; 
    background:rgba(255,255,255,0.1);
    backdrop-filter:blur(10px);
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.2);
    animation:fadeInUp 0.8s ease-out;
'>
    <div style='font-weight:800; margin-bottom:0.8rem; font-size:1.3rem;'>🚀 ResumeMatch AI</div>
    <div style='font-size:1rem; margin-bottom:0.5rem;'>AI-Powered Resume Analysis • Professional Matching • Career Optimization</div>
    <div style='font-size:0.9rem; color:rgba(255,255,255,0.7);'>Built with ❤️ using Streamlit</div>
</div>
""")