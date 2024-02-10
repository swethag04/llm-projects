import streamlit as st
import langchain_helper

st.title('Baby Name Generator')

gender = st.sidebar.selectbox("Choose a gender",
                     ("Girl", "Boy"))
nationality = st.sidebar.selectbox("Choose the nationality", ("American", "Indian", "Chinese", "Russian"))


if gender and nationality:
    response = langchain_helper.generate_baby_names(gender, nationality)
    # st.header(response['baby_names'].strip()).split(",")
    baby_names = response['baby_names'].strip().split(",")
    st.write("** Top 5 Baby Names **")

    for name in baby_names:
        st.write("--", name)