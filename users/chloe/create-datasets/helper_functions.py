from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

def get_tfidf_embeddings(sentences):
    # In the paper, they used TF-IDF as a baseline, removing stopwords, 
    # keeping features appearing in at least 2 documents and e normalized 
    # the TF-IDF product result using the l2-norm

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, norm='l2')

    tfidf_matrix = vectorizer.fit_transform(sentences)

    cols = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=cols)

    return tfidf_df

def get_bert_embeddings(sentences):
    # In the paper, they used a pre-trained BERT model according to the Zero-shot learning paradigm.
    # Each piece of text is converted into a numerical vector of 768 dimensions

    bert_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state    
    sentence_embeddings = last_hidden_states.mean(dim=1)

    return sentence_embeddings.numpy()