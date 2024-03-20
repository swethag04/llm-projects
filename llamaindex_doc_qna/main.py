import os
import streamlit as st
import llamaindex_helper
from tempfile import NamedTemporaryFile

st.title("Q&A App for your PDF document")
st.subheader('Upload your PDF file')
input_file = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                 accept_multiple_files = False, 
                                 type=['pdf'])
if st.button("Process"):
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(input_file.getbuffer())
user_question = st.text_input("Ask a question from the doc")
if st.button("Submit"):
    ans = llamaindex_helper.generate_answer(user_question)
    st.write(ans.response)