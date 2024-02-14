from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os
import key_param


os.environ['OPENAI_API_KEY'] = key_param.openai_api_key

## Setup MongoDB connection
client = MongoClient(key_param.MONGO_URI)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Configure db details
DB_NAME = 'langchain_demo'
db = client[DB_NAME]

COLLECTION_NAME = 'emp-policy'
collection = db[COLLECTION_NAME]

ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Load the data
loader = PyPDFLoader("data/emp-policy.pdf")
data = loader.load()
print(len(data))

# Split the docs and create embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                               chunk_overlap=100)
chunks = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key, disallowed_special=())

# Create embeddings in atlas vector store
vector_search = MongoDBAtlasVectorSearch.from_documents( 
                                documents=chunks, 
                                embedding= embeddings, 
                                collection=collection,
                                index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME 
                                                    )

retriever = vector_search.as_retriever(
                        search_type = "similarity",
                        search_kwargs = {"k": 1}
)
template = """Answer the question: {question} based only on the following context:
context: {context}
 """
prompt = PromptTemplate.from_template(template = template,
                        input_varaibles = ["context", "question"])
output_parser = StrOutputParser()
model = ChatOpenAI(openai_api_key=key_param.openai_api_key, 
             model_name = 'gpt-3.5-turbo',
             temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_chain = (
    {"context": retriever | format_docs,  "question": RunnablePassthrough()}
    | prompt 
    | model 
    | output_parser
)
response = retrieval_chain.invoke("What is the severance pay for an employee during a layoff?")
print(response)
