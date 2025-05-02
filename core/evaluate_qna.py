
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

score_thresholds = {
    'ambiguity': 0.5,
    'incompleteness': 0.5,
    'relevance': 0.4
}

def evaluate_qa_pairs(qa_pairs):
    # data = json.loads(qa_pairs)
    formattedData = [];
    for item in qa_pairs:
        formattedData.append({
            'question': item['question'],
            'answer': item['answer'],
            'confidence_score': 1,
            'relevance_score': 1,
            'flags': ['ambiguous', 'incomplete'],
            'category': item['grc_domain']
        })
    
    return formattedData;

    # Load the pre-trained model for sentence embeddings
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    """Evaluate QA pairs and return with confidence and flags."""
    df_generated = pd.DataFrame(qa_pairs)
    df_generated['question_embeddings'] = df_generated['question'].apply(lambda x: embedding_model.encode(x))
    df_generated['answer_embeddings'] = df_generated['answer'].apply(lambda x: embedding_model.encode(x))

    df_generated = flag_relevance_and_redundancy(df_generated, df_generated['question_embeddings'], df['embeddings'])
    df_generated = mark_unique_or_duplicate(df_generated, threshold=0.9)
    add_ambiguity_score(df_generated)
    add_incompleteness_score(df_generated)
    df_generated['flags'] = df_generated.apply(lambda row: generate_flags(row, score_thresholds), axis=1)
    df_generated['confidence_score'] = df_generated.apply(lambda row: compute_confidence_score(row, weights), axis=1)
        

def flag_relevance_and_redundancy(df, gen_embedding_col, reference_embeddings, relevance_threshold=0.4, redundancy_threshold=0.9):
    # Ensure embeddings are in numpy array format
    gen_embeddings = np.stack(gen_embedding_col)
    ref_embeddings = np.stack(reference_embeddings)

    # Compute cosine similarity matrix: [n_generated, n_reference]
    similarity_matrix = cosine_similarity(gen_embeddings, ref_embeddings)

    # Compute relevance score (max similarity)
    df['relevance_score'] = similarity_matrix.max(axis=1)
    df['relevance_flag'] =  df['relevance_score'] > relevance_threshold  # Flagging irrelevant questions

    return df

def is_ambiguous_question(q):
    # List of vague pronouns or ambiguous terms
    VAGUE_TERMS = ["it", "they", "this", "that", "these", "those", "something", "anything","everything"]
    
    tokens = q.lower().split()
    vague_score = sum(1 for word in tokens if word in VAGUE_TERMS)
    short_score = 1 if len(tokens) < 5 else 0
    vague_ratio = vague_score / len(tokens) if tokens else 1
    return round((vague_ratio + short_score) / 2, 2)  # scaled between 0-1

def add_ambiguity_score(df):
    df['ambiguity_score'] = df['question'].apply(is_ambiguous_question)
    return df

def is_incomplete_answer(ans):
    GENERIC_ANSWERS = ["it depends", "various reasons", "not sure", "unclear", "depends"]
    
    ans_lower = ans.lower()
    if ans_lower.strip() in GENERIC_ANSWERS:
        return 1.0

    tokens = nlp(ans_lower)
    if len(tokens) < 5:
        return 0.9  # too short = likely incomplete

    has_noun = any(token.pos_ in ("NOUN", "PROPN") for token in tokens)
    has_verb = any(token.pos_ == "VERB" for token in tokens)

    if not has_noun or not has_verb:
        return 0.8

    return 0.0  # looks complete

def add_incompleteness_score(df):
    df['incompleteness_score'] = df['answer'].apply(is_incomplete_answer)
    return df

def mark_unique_or_duplicate(df, threshold=0.9):
    """
    Flags each question as 'valid' or 'duplicate' by comparing all pairs
    and only keeping the first valid one in each group of duplicates.
    """
    embeddings = np.stack(df['question_embeddings'])
    similarity = cosine_similarity(embeddings, embeddings)
    np.fill_diagonal(similarity, 0)

    n = len(df)
    status = [False] * n  # default: valid
    seen = set()

    for i in range(n):
        if i in seen:
            status[i] = True
            continue
        dup_indices = np.where(similarity[i] > threshold)[0]
        for j in dup_indices:
            seen.add(j)
            status[j] = True

    df['duplicate_flag'] = status
    return df

def generate_flags(row, thresholds):
    score_thresholds = {
    'ambiguity': 0.5,
    'incompleteness': 0.5,
    'relevance': 0.4
    }
    
    flags = []

    # Ambiguity check
    if row.get('ambiguity_score', 0) > thresholds['ambiguity']:
        flags.append('ambiguous')

    # Incompleteness check
    if row.get('incompleteness_score', 0) > thresholds['incompleteness']:
        flags.append('incomplete')

    # Redundancy check
    if row['duplicate_flag'] is True:
        flags.append('duplicate')

    # # Already exists in DB or reference
    # if row['already_exists'] is True:
    #     flags.append('already_exists')

    # Low relevance (optional)
    if row.get('relevance_score', 1) < thresholds['relevance']:
        flags.append('low_relevance')

    return flags

def compute_confidence_score(row):
    weights = {
    'relevance': 0.3,      # low importance on relevance
    'clarity': 0.5,       # high importance on calarity
    'completeness': 0.45   # high importance on completeness
    }
    
    rel = row.get('relevance_score', 0.0)
    amb = row.get('ambiguity_score', 1.0)
    inc = row.get('incompleteness_score', 1.0)

    score = (
        weights['relevance'] * rel +
        weights['clarity'] * (1 - amb) +
        weights['completeness'] * (1 - inc)
    ) / sum(weights.values())

    return round(max(min(score, 1.0), 0.0), 4)  # Clip to [0, 1] and round
