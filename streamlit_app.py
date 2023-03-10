import streamlit as st
import pdfplumber
import openai
import os

# Authenticate with the OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Define the function to extract text from a PDF file
def extract_text(pdf_file):
    with pdfplumber.load(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Define the function to generate a response using the GPT-3 API
def generate_response (prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7, 
        max_tokens=2050,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return completions.choices[0].text.strip()

# Define the Streamlit app
def app():
    st.title('PDF GPT-3')

    # Upload the PDF file
    pdf_file = st.file_uploader('Upload a PDF file')

    if pdf_file is not None:
        # Convert the PDF file to text
        text = extract_text(pdf_file)

        # Take user input for the question
        question = st.text_input('Ask a question')

        if question:
            # Generate the response using the GPT-3 API
            model_engine = "text-davinci-003"
            prompt = f"What is the answer to the following question about the PDF document: {question}\nPDF text: {text}\nAnswer:"
            response = generate_response(prompt, model_engine)

            # Display the response
            st.text_area('Response', response)
