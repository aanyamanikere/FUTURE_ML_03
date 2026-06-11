# FUTURE_ML_03
# AI Resume Screening & Candidate Screening System 

## Project Objective
This project is an automated tool that helps make hiring easier. It uses Natural Language Processing (NLP) to read and understand candidate resumes, identify important skills, and compare them with the requirements listed in job descriptions.

By checking profiles, the engine generates a ranking system that identifies matches and instantly highlights critical skill gaps for recruiters.

---

## Tech Stack & Libraries
- **Language:** Python
- **Environment:** VS Code / GitHub Codespaces
- **Core Processing:** Pandas, NumPy
- **NLP & Text Vectorization:** Scikit-learn (TF-IDF Vectorization, Cosine Similarity calculations)

---

## Key System Components

### 1. Resume Text Preprocessing
Before analyzing a resume, the system cleans and prepares the text by:
- Converting all text to lowercase.
- Removing unnecessary characters, extra spaces and formatting issues.
- Filtering out common words that do not add much value, so that important technical skills and keywords can be identified more easily.

### 2. Feature Extraction & Skill Context Mapping
Instead of simply looking for exact keywords, the system uses TF-IDF (Term Frequency–Inverse Document Frequency) to measure how important specific skills are in a resume. This helps identify valuable skills such as Python, TensorFlow, AWS, and Financial Modeling by considering how frequently they appear in a candidate's resume compared to other resumes.

### 3. Match Scoring Engine
The system uses Cosine Similarity to compare a candidate's resume with a job description. It measures how closely the candidate's skills and experience match the job requirements. The similarity score ranges from 0.0 (no match) to 1.0 (perfect match), helping rank candidates based on how well they fit the position.

---

## Operational Validation & Output

### Automated Match Scoring Simulation
```text
[INSERT YOUR CODESPACE TERMINAL OUTPUT HERE]
