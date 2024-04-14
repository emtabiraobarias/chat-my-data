from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

import glob

print("Loading data...")
index_path = "data/json/index/*_index.txt"
for index_file in glob.glob(index_path):
    print(index_file)
    loader = UnstructuredFileLoader(index_file)
    raw_documents = loader.load()


print("Splitting text...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)
documents = text_splitter.split_documents(raw_documents)


#TODO: Address chunking of data for a more efficient processing
print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("cvindex_vectorstore")
