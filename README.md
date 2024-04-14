# query-cv-index
Personal project following langchain tutorial on https://blog.langchain.dev/tutorial-chatgpt-over-your-data/ - credits due to the tutorial blog's author for some of the python script files used in this personal project.

[x] Summarise single CV pdf file by extracting key information from the document and dumping to a JSON file.

[x] Bulk process documents

[x] Save to vector store

[ ] Cluster common applicants or Data Cleansing https://github.com/mendableai/QA_clustering/blob/main/notebooks/clustering_approach.ipynb || https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625 

[ ] Query with RAG https://python.langchain.com/docs/use_cases/question_answering/

[ ] Create a single cli_app to run everything

# Applied instructions copied from tutorial

## Chat-Your-Data

Create a ChatGPT like experience over your custom docs using [LangChain](https://github.com/langchain-ai/langchain).


### Step 0: Install requirements

`pip install -r requirements.txt`

### Step 1: Set your open AI Key

```sh
export OPENAI_API_KEY=<your_key_here>
```


### Step 2: Ingest your data

Run: `python summarise_data.py`

Run: `python ingest_data.py`

This builds `cvindex_vectorstore` files using OpenAI Embeddings and FAISS.

### Query data

Custom prompts are used to ground the answers in the CV index file.

### Running the Application

By running `python app.py` from the command line you can easily interact with your ChatGPT over the CV index data.

You may also run `cli_app.py` from the command line to query CV index data over the command line. You will have to familiarise yourself to the options by looking inside the python file.

# Tutorials that helped
* https://blog.langchain.dev/tutorial-chatgpt-over-your-data/

* https://www.gettingstarted.ai/how-to-extract-metadata-from-pdf-convert-to-json-langchain/

* https://medium.com/@larry_nguyen/langchain-101-lesson-3-output-parser-406591b094d7