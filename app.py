import streamlit as st
import PyPDF2
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Career Recommendation", layout="wide")

st.title("📄 AI Resume-Based Career Recommendation System")

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    st.subheader("🔍 Extracted Resume Text")
    st.write(resume_text[:800])

    skills_db = ["python","machine learning","data analysis","sql","deep learning","nlp","excel","java"]
    found_skills = [skill for skill in skills_db if skill in resume_text.lower()]

    st.subheader("💡 Detected Skills")
    st.write(found_skills)

    jobs = pd.read_csv("jobs.csv")
    job_desc = jobs["description"].tolist()

    resume_embedding = model.encode([resume_text])
    job_embeddings = model.encode(job_desc)

    scores = cosine_similarity(resume_embedding, job_embeddings)[0]
    jobs["Match Score"] = scores

    st.subheader("🎯 Job Matches")
    st.dataframe(jobs.sort_values(by="Match Score", ascending=False))

    best_job = jobs.sort_values(by="Match Score", ascending=False).iloc[0]
    job_skills = best_job["skills"].split(", ")
    missing_skills = [skill for skill in job_skills if skill not in found_skills]

    st.subheader("📉 Skill Gap Analysis")
    st.write(missing_skills)

    st.subheader("📚 Re-Education Roadmap")
    for skill in missing_skills:
        st.write(f"• Learn {skill} through online courses and projects")
