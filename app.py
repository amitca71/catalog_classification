from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import stanza
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)


architect_descriptions_list = []
catalog_items_list = []
similarities_list=[]

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
        global similarities_list
        global architect_descriptions_list
        global catalog_items_list
        similarities = cosine_similarity(arch_vectors, catalog_vectors)
        similarities_list=similarities.tolist()
        architect_descriptions_list=arch_df['description'].tolist()
        catalog_items_list=catalog_df['item'].tolist()
        # Store data in session
#        session['architect_descriptions'] = arch_df['description'].tolist()
#        session['catalog_items'] = catalog_df['item'].tolist()
#        session['similarities'] = similarities.tolist()
        
        # Redirect to results page
        return redirect(url_for('results'))
    
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
#    if 'architect_descriptions' not in session:
#        return redirect(url_for('index'))
    
#    descriptions = session['architect_descriptions']
    global architect_descriptions_list
    return render_template('results.html', descriptions=architect_descriptions_list)

@app.route('/get_matches', methods=['POST'])
def get_matches():
    data = request.json
    idx = data['index']
    n = data.get('n', 5)
    similarities=np.array(similarities_list)
#    similarities = np.array(session['similarities'])
#    catalog_items = session['catalog_items']
    
    # Get top N indices
    top_indices = similarities[idx].argsort()[-n:][::-1]
    matches = [catalog_items_list[i] for i in top_indices]
    
    return jsonify(matches=matches)

if __name__ == '__main__':
    app.run(debug=True)