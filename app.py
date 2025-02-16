from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import stanza
import numpy as np
import os

app = Flask(_name_)
app.secret_key = os.urandom(24)

# Download Hebrew model for Stanza
stanza.download('he')
nlp = stanza.Pipeline(lang='he', processors='tokenize,lemma')

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    return ' '.join(lemmas)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Read uploaded files
        arch_file = request.files['arch_file']
        catalog_file = request.files['catalog_file']
        
        # Read CSV data
        arch_df = pd.read_csv(arch_file, encoding='utf-8')
        catalog_df = pd.read_csv(catalog_file, encoding='utf-8')
        
        # Preprocess texts
        arch_texts = arch_df['description'].apply(lemmatize_text).tolist()
        catalog_texts = catalog_df['item'].apply(lemmatize_text).tolist()
        
        # Combine texts for TF-IDF
        combined_texts = arch_texts + catalog_texts
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)
        
        # Split back to arch and catalog
        arch_vectors = tfidf_matrix[:len(arch_texts)]
        catalog_vectors = tfidf_matrix[len(arch_texts):]
        
        # Compute similarities
        similarities = cosine_similarity(arch_vectors, catalog_vectors)
        
        # Store data in session
        session['architect_descriptions'] = arch_df['description'].tolist()
        session['catalog_items'] = catalog_df['item'].tolist()
        session['similarities'] = similarities.tolist()
        
        return render_template('results.html', 
                            descriptions=session['architect_descriptions'])
    
    return render_template('index.html')

@app.route('/get_matches', methods=['POST'])
def get_matches():
    data = request.json
    idx = data['index']
    n = data.get('n', 5)
    
    similarities = np.array(session['similarities'])
    catalog_items = session['catalog_items']
    
    # Get top N indices
    top_indices = similarities[idx].argsort()[-n:][::-1]
    matches = [catalog_items[i] for i in top_indices]
    
    return jsonify(matches=matches)

if _name_ == '_main_':
    app.run(debug=True)
