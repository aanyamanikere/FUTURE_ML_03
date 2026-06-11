import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. SAMPLE UNSTRUCTURED RESUME DATASET (Matches Kaggle Structure)
# =====================================================================
# Real dataset loading setup:
# df = pd.read_csv("data/Resume.csv")
resumes_mock = {
    "candidate_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"],
    "resume_text": [
        "Data Scientist with 3 years experience. Proficient in Python, machine learning, SQL, pandas, and scikit-learn. Built time series forecasting models.",
        "Senior Software Engineer. Expert in Java, C++, cloud architecture, microservices, and system design. Extensive background in Spring Boot and Docker.",
        "Data Analyst and Python programmer. Skilled in data cleaning, SQL visualization, Power BI, and basic machine learning implementations.",
        "HR Manager and Talent Acquisition Specialist. Experienced in recruitment, payroll onboarding, employee relations, and strategic workforce planning."
    ]
}
df = pd.DataFrame(resumes_mock)

# TARGET JOB DESCRIPTION TO EVALUATE AGAINST
TARGET_JOB_DESCRIPTION = """
Seeking a Data Scientist / Machine Learning Engineer with strong proficiency in 
Python, SQL, machine learning architectures, pandas, and scikit-learn data pipelines. 
Experience with analytical text modeling or predictive systems is highly preferred.
"""

print("=====================================================================")
print("            INITIALIZING RESUME SCREENING ENGINE                    ")
print("=====================================================================")
print(f"Target Role Criteria Profile:\n{TARGET_JOB_DESCRIPTION.strip()}\n")

# =====================================================================
# 2. RAW TEXT PREPROCESSING & STANDARDIZATION
# =====================================================================
def clean_document_text(text):
    text = text.lower()
    tokens = [word for word in text.split() if word.isalnum()]
    return " ".join(tokens)

df['cleaned_resume'] = df['resume_text'].apply(clean_document_text)
cleaned_job_desc = clean_document_text(TARGET_JOB_DESCRIPTION)

# =====================================================================
# 3. FEATURE EXTRACTION (TF-IDF TOKEN WEIGHTING)
# =====================================================================
# Initialize vectorizer with standard English stop words removed
vectorizer = TfidfVectorizer(stop_words='english')

# Combine all texts to build a comprehensive vocabulary matrix space
all_documents = list(df['cleaned_resume']) + [cleaned_job_desc]
tfidf_matrix = vectorizer.fit_transform(all_documents)

# Separate the historical resumes matrix from the target job profile vector
resume_vectors = tfidf_matrix[:-1]
job_desc_vector = tfidf_matrix[-1]

# =====================================================================
# 4. SIMILARITY GEOMETRY & RANKING ENGINE
# =====================================================================
# Compute cosine similarity between every resume and the job specification
similarity_scores = cosine_similarity(resume_vectors, job_desc_vector).flatten()

# Inject analytical values into our structured data dashboard
df['match_score'] = similarity_scores
df['match_percentage'] = np.round(similarity_scores * 100, 2)

# Rank candidates chronologically based on vector closeness
df['rank'] = df['match_percentage'].rank(ascending=False, method='min').astype(int)
ranked_candidates = df.sort_values(by='rank').reset_index(drop=True)

# =====================================================================
# 5. AUTOMATED GAP ANALYSIS (Optional Bonus Element)
# =====================================================================
job_terms = set(vectorizer.inverse_transform(job_desc_vector)[0])

def evaluate_skill_gaps(resume_vec, job_vocab):
    resume_terms = set(vectorizer.inverse_transform(resume_vec)[0])
    missing_critical_skills = job_vocab.difference(resume_terms)
    return ", ".join(list(missing_critical_skills)[:4]) # Display up to 4 key terms

gaps = []
for idx in range(len(df)):
    gaps.append(evaluate_skill_gaps(resume_vectors[idx], job_terms))
df['identified_gaps'] = gaps

# Refresh sorting configuration post-gap integration
ranked_candidates = df.sort_values(by='rank').reset_index(drop=True)

# =====================================================================
# 6. SYSTEM EVALUATION & METRICS DISPLAY
# =====================================================================
print("=====================================================================")
print("                   CANDIDATE RANKING DASHBOARD                       ")
print("=====================================================================")
for i, row in ranked_candidates.iterrows():
    print(f"Rank {row['rank']}: {row['candidate_name']}")
    print(f"Match Score: {row['match_percentage']}%")
    print(f"Key Missing Tokens / Gaps: [{row['identified_gaps']}]")
    print("-" * 69)