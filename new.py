
import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from pathlib import Path
import pickle

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def load_courses():
    df = pd.read_csv("wm_courses_2025.csv")
    df.dropna(subset=['course_title', 'course_description'], inplace=True)
    df['course_level'] = df['course_code'].apply(extract_course_level)
    return df

def extract_course_level(course_code):
    match = re.search(r'(\d{3,4})', str(course_code))
    if match:
        number = int(match.group(1))
        if number < 200:
            return 100
        elif number < 300:
            return 200
        elif number < 400:
            return 300
        else:
            return 400
    return None

def calculate_level_bonus(course_level, target_level):
    if course_level is None or target_level is None:
        return 0.0
    level_diff = abs(course_level - target_level)
    if level_diff == 0:
        return 0.15
    elif level_diff == 100:
        return 0.12
    elif level_diff == 200:
        return 0.02
    else:
        return 0.0

def calculate_transferability(title1, desc1, title2, desc2):
    try:
        emb_desc = model.encode([desc1, desc2])
        sim_desc = cosine_similarity([emb_desc[0]], [emb_desc[1]])[0][0]
        emb_title = model.encode([title1, title2])
        sim_title = cosine_similarity([emb_title[0]], [emb_title[1]])[0][0]
        score = 1 / (1 + math.exp(-(-7.144 + 9.219 * sim_desc + 5.141 * sim_title)))
        return sim_desc, sim_title, score
    except:
        return None, None, None

def get_transferability_category(score):
    if score >= 0.85:
        return "Very High", "🟢"
    elif score >= 0.7279793:
        return "Likely Transferable", "🔵"
    elif score >= 0.6:
        return "Possibly Transferable", "🟡"
    elif score >= 0.4:
        return "Unlikely Transferable", "🟠"
    else:
        return "Very Low", "🔴"

st.title("📘 Course Transferability Analyzer")

st.markdown("Upload and analyze external courses against William & Mary courses.")

df = load_courses()
st.success(f"Loaded {len(df)} W&M courses")

num_courses = st.slider("Number of external courses", 1, 5, 1)

results = []

for i in range(num_courses):
    st.subheader(f"External Course {i+1}")
    title = st.text_input(f"Course {i+1} Title")
    desc = st.text_area(f"Course {i+1} Description")
    level = st.selectbox(f"Target Level {i+1}", [None, 100, 200, 300, 400], format_func=lambda x: "Any" if x is None else f"{x} Level")
    keywords = st.text_input(f"Keywords {i+1} (optional)")

    if title and desc:
        ext_text = f"{title} {desc}"
        ext_emb = model.encode([ext_text])

        filtered = df.copy()
        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.split(",")]
            filtered = filtered[filtered.apply(lambda row: any(k in (str(row['course_code']) + " " + row['course_title'] + " " + row['course_description']).lower() for k in keyword_list), axis=1)]

        filtered_texts = filtered.apply(lambda row: f"{row['course_code']} {row['course_title']} {row['course_description']}", axis=1).tolist()
        filtered_embs = model.encode(filtered_texts)

        sims = cosine_similarity(ext_emb, filtered_embs)[0]

        if level is not None:
            bonuses = filtered['course_level'].apply(lambda x: calculate_level_bonus(x, level)).tolist()
            sims = [s + b for s, b in zip(sims, bonuses)]

        top_idx = np.argsort(sims)[-5:][::-1]

        for idx in top_idx:
            row = filtered.iloc[idx]
            sim_desc, sim_title, score = calculate_transferability(title, desc, row['course_title'], row['course_description'])
            category, emoji = get_transferability_category(score)

            st.markdown(f"**{row['course_code']} - {row['course_title']}** (Level {row['course_level']})")
            st.markdown(f"- Adjusted Similarity: {sims[idx]:.4f}")
            st.markdown(f"- Description Similarity: {sim_desc:.4f}")
            st.markdown(f"- Title Similarity: {sim_title:.4f}")
            st.markdown(f"- **Transferability Score:** {score:.4f} {emoji} ({category})")
            st.markdown("---")
