import streamlit as st
from io import StringIO
from pypdf import PdfReader
#from PyPDF2 import PdfReader
from app import parse_document, parse_url

files= st.file_uploader(label="upload a file", accept_multiple_files=True, type="pdf")
st.write("if you prefer to interact with a particular URL such as a docs e.g https://docs.python.org,\n")
url = st.text_input("provide the URL in the field below, then press the enter key")
st.write(url)

# st.write(file_upload[0]._file_urls.upload_url)

all_docs = []
docs = ""
input_text = st.text_input(label="Ask your documents/url a question:")
# st.write(input_text)
    
pressed = st.button(label="Get Response", type="primary")

user_query = "inurl: " + url + " " + input_text

if len(url) > 0 and pressed is True:
    #st.write(url)
    #input_text

    response = parse_url(user_query)
    
    if 'response' not in st.session_state:
        st.session_state['response'] = response

    st.write(st.session_state.response)

else:
    try:        
        for file in files:
            if file is not None:
                file_data = PdfReader(file)
                # extract text from the pdf file
                for page in file_data.pages:
                    docs += page.extract_text()
            #all_docs.append(docs)
            if len(input_text) > 0:
                response = parse_document(docs=docs, question= input_text)
                if 'file_response' not in st.session_state:
                    st.session_state['file_response'] = response
                    st.write(st.session_state.file_response)
            else:
                st.write("Ask a question")
    except:
        st.write("No answer")
    
