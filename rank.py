import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

encoding = detect_encoding('cvs/PERSON1_CV.txt')
print(f"Detected Encoding: {encoding}")

# Step 1: Load Job Descriptions and Candidate CVs
def load_data(job_description_path, cv_folder_path):
    # Load job description
    with open(job_description_path, 'r', encoding='utf-8') as file:
        job_description = file.read()
    print("Job Description Loaded Successfully!")
    print(f"Job Description: {job_description}")  # Debug: Print the job description

    # Load candidate CVs
    candidate_cvs = {}
    for filename in os.listdir(cv_folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(cv_folder_path, filename), 'r') as file:
                candidate_cvs[filename] = file.read()
            print(f"Loaded CV: {filename}")  # Debug: Print the filename of each loaded CV

    print(f"Candidate CVs: {candidate_cvs}")  # Debug: Print the loaded CVs
    return job_description, candidate_cvs

# Step 2: Extract Keywords using TF-IDF
def extract_keywords(text, top_n=20):  # Increased top_n to 20
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Get top N keywords
    top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-top_n:][::-1]]
    print(f"Top Keywords: {top_keywords}")  # Debug: Print the top keywords
    return top_keywords

# Step 3: Match Keywords with Candidate CVs
def match_candidates(job_description, candidate_cvs):
    # Extract keywords from job description
    job_keywords = extract_keywords(job_description)
    print(f"Job Keywords: {job_keywords}")  # Debug: Print job keywords

    # Compute similarity scores for each candidate
    similarity_scores = {}
    for candidate_name, cv_text in candidate_cvs.items():
        candidate_keywords = extract_keywords(cv_text)
        print(f"Candidate Keywords for {candidate_name}: {candidate_keywords}")  # Debug: Print candidate keywords
        common_keywords = set(job_keywords).intersection(set(candidate_keywords))
        similarity_score = len(common_keywords) / len(job_keywords)
        similarity_scores[candidate_name] = similarity_score
        print(f"Candidate: {candidate_name}, Similarity Score: {similarity_score}")  # Debug: Print similarity score

    return similarity_scores

# Step 4: Handle Missing Values
def handle_missing_values(similarity_scores, default_score=0.5):
    for candidate_name, score in similarity_scores.items():
        if pd.isna(score) or score == 0:  # Handle zero scores as well
            similarity_scores[candidate_name] = default_score
    return similarity_scores

# Step 5: Rank Candidates
def rank_candidates(similarity_scores):
    ranked_candidates = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Main Function
def main(job_description_path, cv_folder_path):
    # Load data
    job_description, candidate_cvs = load_data(job_description_path, cv_folder_path)

    # Match candidates
    similarity_scores = match_candidates(job_description, candidate_cvs)

    # Handle missing values
    similarity_scores = handle_missing_values(similarity_scores)

    # Rank candidates
    ranked_candidates = rank_candidates(similarity_scores)

    # Output results
    print("Ranked Candidates:")
    for candidate, score in ranked_candidates:
        print(f"{candidate}: {score:.2f}")

# Run the program
if __name__ == "__main__":
    job_description_path = 'C:/Users/khush/OneDrive\Documents/ai_job/job_desc/job1.txt' # Path to job description file
    cv_folder_path = "cvs"  # Folder containing candidate CVs in txt format
    main(job_description_path, cv_folder_path)