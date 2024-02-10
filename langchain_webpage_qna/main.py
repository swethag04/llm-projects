import streamlit as st
import rag_langchain_helper


st.title("Q&A App for your website")
web_input = st.text_input("Enter a website link")
user_question = st.text_input("Ask a question from the website")
st.button("Submit")
ans = rag_langchain_helper.gen_answer(web_input, user_question)
st.write(ans)





