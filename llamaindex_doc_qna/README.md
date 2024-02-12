# Research paper Q&A App using Retrieval Augmented Generation (RAG)

### Goal 
The goal of the project is to build a Question and Answer system powered by Large Language Models (LLM) that will answer any specific question from a research paper the user uploads on the app.
<br>

### Data Problem
The data task is to use Retreival Augment Generation  technique to provide additional context to LLM to answer a question.
I will be testing 2 different embedding models as well as 2 different LLMs to compare their performances to identify the best model to use for the app.

### Data Required
Digital copies of research papers in pdf format
<br>

### How to source the data?
The data can be sourced by scraping the internet for popular research papers
<br>


### Expected Results
The app should reliably and correctly answer the questions using the information from the specific document.
<br> 

### Technology and Tools Used

1. Large Language Models (LLMs) <br>
   LLMS are deep learning models that are pre-trained on vast amounts of data and can be used for tasks like generating text, summarizing, translating, answering 
   questions etc.
    <br>
2. Retrieval Augmented Generation (RAG) <br>
   RAG is a technique that enhances the capabilties of LLMs by incorporating information retrieval into the text generation process. This is done by retrieving 
   data/documents relevant to a question or task and providing them as context for the LLM.
   <br>
3. LlamaIndex <br>
   LlamaIndex is an open source data framework for building RAG systems.
   <br>
4. Streamlit <br>
   Streamlit is an open source python library used to create custom web apps for ML
   <br>



### Results
1. Embedding models comparison:
   Based on the evaluation results, context_precision and context_recall metrics of the OpenAI embedding model are the same as Bge embedding model in my RAG pipeline when applied to my own dataset.
   Hence, either of the two embedding models can be used for building the app. 

2. LLM comparison: <br>
       Ragas Score for Zephyr model = 0.8887 <br>
       Ragas Score for Falcon model = 0.8563 <br>

The Zephyr model seems to slightly outperform the Falcon model in my RAG pipeline when applied to my own dataset and hence might be a better choice to use for the Q&A App

### Future Work:
1. Compare different chunk sizes to see how it impacts the model performance
2. Increase test data size to see how that impacts the metrics
3. Consider evalutaing using other metrics like BEIR to evaluate retreival
