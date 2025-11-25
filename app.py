# app.py - AI Resume Matcher Pro (FIXED UI + Visible Colors)
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
# Page config - FIXED FOR VISIBILITY
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

# ----------------------------
# FIXED CSS - LIGHT COLORS + PROPER VISIBILITY
# ----------------------------
st.markdown(
    """
<style>
/* ===== FIX STREAMLIT DEFAULT SPACING ===== */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ===== CLEAN WHITE BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ===== FIXED HEADER - LIGHT & VISIBLE ===== */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    padding: 2rem 2.5rem !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.15) !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
    border: 1px solid #e2e8f0 !important;
}

.app-main-title {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: white !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    margin: 0.5rem 0 0 0 !important;
}

/* ===== CLEAN WHITE CARDS ===== */
.clean-card {
    background: white !important;
    padding: 2rem !important;
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    margin-bottom: 1.5rem !important;
    transition: all 0.3s ease !important;
}

.clean-card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12) !important;
    transform: translateY(-2px) !important;
}

/* ===== IMPROVED BUTTONS - VISIBLE COLORS ===== */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

/* ===== CLEAR INPUT FIELDS ===== */
.stTextArea textarea {
    background: #f8fafc !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: 2px solid #e2e8f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    transition: all 0.3s ease !important;
    color: #1e293b !important;
}

.stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    background: white !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
}

/* ===== VISIBLE SKILL TAGS ===== */
.skill-tag {
    display: inline-block !important;
    margin: 0.3rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: 20px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s ease !important;
}

.skill-tag:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
}

/* ===== STEP INDICATOR - VISIBLE ===== */
.step-indicator {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 2rem 0 !important;
    position: relative !important;
}

.step {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    flex: 1 !important;
    position: relative !important;
}

.step-number {
    width: 45px !important;
    height: 45px !important;
    border-radius: 50% !important;
    background: #f1f5f9 !important;
    color: #64748b !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    z-index: 2 !important;
    transition: all 0.3s ease !important;
    border: 2px solid #e2e8f0 !important;
}

.step.active .step-number {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
    border-color: #4f46e5 !important;
}

.step-line {
    position: absolute !important;
    top: 22px !important;
    left: 50% !important;
    right: -50% !important;
    height: 3px !important;
    background: #e2e8f0 !important;
    z-index: 1 !important;
}

.step.active .step-line {
    background: #4f46e5 !important;
}

.step-label {
    font-size: 0.9rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    text-align: center !important;
}

.step.active .step-label {
    color: #4f46e5 !important;
    font-weight: 600 !important;
}

/* ===== TAB STYLING ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem !important;
    background-color: #f8fafc !important;
    padding: 0.5rem !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 50px !important;
    background-color: #f8fafc !important;
    border-radius: 8px !important;
    padding: 0 1rem !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background-color: white !important;
    color: #4f46e5 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    border-radius: 10px !important;
}

/* ===== MOBILE RESPONSIVENESS ===== */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
    }
    
    .app-header {
        padding: 1.5rem 1rem !important;
    }
    
    .app-main-title {
        font-size: 1.8rem !important;
    }
    
    .app-subtitle {
        font-size: 1rem !important;
    }
    
    .clean-card {
        padding: 1.5rem !important;
    }
    
    .step-indicator {
        flex-wrap: wrap !important;
        gap: 1rem !important;
    }
    
    .step {
        flex: 0 0 calc(50% - 1rem) !important;
        margin-bottom: 1rem !important;
    }
    
    .step-line {
        display: none !important;
    }
}

/* ===== HIDE STREAMLIT DEFAULT ELEMENTS ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== BETTER METRICS ===== */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
}

/* ===== SUCCESS/ERROR/WARNING MESSAGES ===== */
.stSuccess {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
}

.stError {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
}

.stWarning {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
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
# FIXED UI BUILDING
# ----------------------------

# FIXED HEADER - Now clearly visible
html(f"""
<div class="app-header">
    <div class="app-main-title">🚀 ResumeMatch AI</div>
    <div class="app-subtitle">Smart Resume Analysis • AI-Powered Matching • Professional Reports</div>
</div>
""")

# Step Indicator
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

# Enhanced Sidebar
with st.sidebar:
    html('<div class="clean-card">')
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

# Main Content Area with Tabs
tab1, tab2, tab3 = st.tabs(["📝 Resume Input", "💼 Job Analysis", "📊 Match Results"])

with tab1:
    html('<div class="clean-card">')
    st.subheader("📄 Your Resume Content")
    
    # File upload section
    col1, col2 = st.columns([1, 1])
    with col1:
        upload_mode = st.radio(
            "Choose input method:",
            ["📋 Paste Text", "📁 Upload PDF"],
            horizontal=True
        )
    
    if upload_mode == "📁 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload your resume PDF",
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
                        with st.expander("👀 Preview extracted text"):
                            st.text_area(
                                "Extracted Content", 
                                value=st.session_state["resume_text"][:1000] + "..." if len(st.session_state["resume_text"]) > 1000 else st.session_state["resume_text"],
                                height=200,
                                key="preview_area"
                            )
    
    # Text area for resume
    st.session_state["resume_text"] = st.text_area(
        "Paste or edit your resume content:",
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
        key="resume_area"
    )
    
    # Character count
    if st.session_state["resume_text"]:
        char_count = len(st.session_state["resume_text"])
        word_count = len(st.session_state["resume_text"].split())
        st.caption(f"📊 {char_count} characters, {word_count} words")
    
    html('</div>')
    
    # AI Improvement Section
    if use_ai_rewrite:
        html('<div class="clean-card">')
        st.subheader("🤖 AI Resume Improver")
        
        target_role = st.text_input(
            "Target role (optional):",
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
    html('<div class="clean-card">')
    st.subheader("💼 Job Description Analysis")
    
    st.session_state["job_desc_input"] = st.text_area(
        "Paste the job description:",
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
                with st.expander("🔧 Technical Skills Breakdown"):
                    for cat, skills in jd_struct["job_tech"].items():
                        if skills:
                            st.write(f"**{cat}:**")
                            cols = st.columns(3)
                            for i, skill in enumerate(skills):
                                cols[i % 3].markdown(f"<span class='skill-tag'>{skill}</span>", unsafe_allow_html=True)
                
                if jd_struct["job_soft"]:
                    with st.expander("💬 Soft Skills Required"):
                        cols = st.columns(3)
                        for i, skill in enumerate(jd_struct["job_soft"]):
                            cols[i % 3].markdown(f"<span class='skill-tag'>{skill}</span>", unsafe_allow_html=True)
    
    html('</div>')

with tab3:
    html('<div class="clean-card">')
    st.subheader("🚀 Ready to Analyze!")
    
    st.info("""
    **Next Steps:**
    1. ✅ Add your resume in the first tab
    2. ✅ Add job description in the second tab  
    3. 🎯 Click the button below to get your match analysis
    4. 📥 Download your personalized report
    """)
    
    # Central analyze button
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
                html('<div class="clean-card">')
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
                    html('<div class="clean-card">')
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
                    html('<div class="clean-card">')
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
                        marker_color='#4f46e5'
                    ))
                    fig.add_trace(go.Bar(
                        name="Required Skills", 
                        x=cats, 
                        y=job_counts,
                        marker_color='#7c3aed'
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
                html('<div class="clean-card">')
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

# Enhanced Footer
html("""
<div style='text-align:center; color:#64748b; margin-top:3rem; padding:2rem; background:white; border-radius:16px; border:1px solid #e2e8f0;'>
    <div style='font-weight:700; margin-bottom:0.5rem; font-size:1.1rem;'>🚀 ResumeMatch AI</div>
    <div style='font-size:0.95rem;'>AI-Powered Resume Analysis • Professional Matching • Career Optimization</div>
    <div style='font-size:0.85rem; margin-top:0.8rem; color:#94a3b8;'>Built with ❤️ using Streamlit</div>
</div>
""")