import streamlit as st
import pdfminer
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize
from numpy import dot
from numpy.linalg import norm

# Load the PDF GPT model
model = load_model('pdf_gpt.h5')

# Define the function to generate embeddings
def generate_embeddings(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Load the tokenizer and embeddings model
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    embeddings_model = tf.keras.models.load_model('dan.h5')

    # Generate embeddings for each sentence
    embeddings = []
    for sentence in sentences:
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        embedding = embeddings_model.predict([sequence])[0]
        embeddings.append(embedding)

    return embeddings

# Define the function to perform semantic search
def semantic_search(query, embeddings, threshold=0.5):
    # Generate the query embedding
    query_embedding = generate_embeddings(query)[0]

    # Find the most similar sentence to the query
    similarities = []
    for embedding in embeddings:
        similarity = dot(embedding, query_embedding)/(norm(embedding)*norm(query_embedding))
        similarities.append(similarity)

    # Select the sentences with similarity greater than the threshold
    relevant_indices = [i for i, similarity in enumerate(similarities) if similarity > threshold]

    return relevant_indices

# Define the function to generate responses
def generate_responses(query, text):
    # Split the text into chunks of 100 sentences
    sentences = sent_tokenize(text)
    chunks = [sentences[i:i+100] for i in range(0, len(sentences), 100)]

    # Generate embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        embeddings.extend(generate_embeddings(' '.join(chunk)))

    # Perform semantic search on the query
    relevant_indices = semantic_search(query, embeddings)

    # Generate the response using the relevant sentences
    response = ''
    for i in relevant_indices:
        sentence = chunks[i//100][i%100]
        page_number = 'Page ' + str(sentences.index(sentence)//50 + 1)
        response += sentence + ' [' + page_number + ']\n'

    return response

# Define the Streamlit app
def app():
    st.title('PDF GPT')

    # Upload the PDF file
    pdf_file = st.file_uploader('Upload a PDF file')

    if pdf_file is not None:
        # Convert the PDF file to text
        text = extract_text(pdf_file)

        # Take user input for the query
        query = st.text_input('Ask a question')

        if query:
            # Generate the response
            response = generate_responses(query, text)

            # Display the response
            st.text_area('Response', response)
