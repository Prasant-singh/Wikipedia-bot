import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


nltk.download('punkt')

def get_page_content(url):

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])
    return content

def preprocess_content(content):
    
    sentences = nltk.sent_tokenize(content)
    return sentences

def answer_query(query, sentences):
    
    vectorizer = TfidfVectorizer().fit_transform(sentences + [query])
    vectors = vectorizer.toarray()

    
    query_vector = vectors[-1].reshape(1, -1)  
    sentence_vectors = vectors[:-1]  
    
    cosine_similarities = cosine_similarity(query_vector, sentence_vectors)
    similar_sentence_idx = np.argmax(cosine_similarities)
    
  
    return sentences[similar_sentence_idx]

from transformers import pipeline


qa_pipeline = pipeline("question-answering")

def answer_query(query, content):
 
    result = qa_pipeline(question=query, context=content)
    return result['answer']


def bot():
    url = input("Enter the Wikipedia URL: ")
    
   
    content = get_page_content(url)
    
    print("You can now ask questions based on the content of the URL.")
    
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break

       
        answer = answer_query(query, content)
        print("Bot:", answer)
        
bot()
