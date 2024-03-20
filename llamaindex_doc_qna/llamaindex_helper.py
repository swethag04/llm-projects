import os
from secretkey import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

import openai
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
# Print the API key for verification
# print(f"API key: {api_key}")

from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.llms import HuggingFaceInferenceAPI

import nest_asyncio
nest_asyncio.apply()

def load_docs():
    """
    Load a pdf document that will provide additional context to LLM
    """
    docs = SimpleDirectoryReader(".").load_data()
    return docs

def get_vector_store(docs, zephyr_llm):
    """
        Split document into smaller sized chunks for embedding
        and store document chunks in a vector store
    """
    index = VectorStoreIndex.from_documents(docs,
                                        service_context=ServiceContext.from_defaults(chunk_size=512),
                                        llm = zephyr_llm)
    return index

def build_query_engine(index):
    query_engine = index.as_query_engine()
    return query_engine

def generate_answer(question):  
    zephyr_llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha",
                                        token="")
    docs = load_docs()
    index = get_vector_store(docs, zephyr_llm)
    qe = build_query_engine(index)
    response = qe.query(question)
    return response
