# Q&A App for a Website


This project is a streamlit app to answer questions from a website. The user inputs a website link and a question related to the content in the website and the app generates an appropriate response.

The app uses langchain modules to get all the contents from the webpage, convert the data into embeddings using OpenAI embedding model, store the embeddings in a  Chroma DB vector database, and retrieve the answer based on the question provided by the user using OpenAI gpt-3.5 model.


![Screenshot of streamlit website an app output](https://github.com/swethag04/llm/blob/main/website-qna-app/qna-app.png)
