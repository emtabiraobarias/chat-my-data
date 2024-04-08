from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def load_pdf(pdf_file_path="data/test_cv.pdf"):
    print("Loading data...")
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    return pages

def define_llm_chain(model_name="gpt-4", custom_prompt=""):
    print("Defining LLM chain")
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return llm_chain

def stuff_summarise_pdf(pdf_file_path, model_name, custom_prompt, document_variable_name):
    print("Stuff summarising PDF content")
    pages = load_pdf(pdf_file_path)
    llm_chain = define_llm_chain(model_name, custom_prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name=document_variable_name)
    summary = stuff_chain.run(pages)
    
    return summary

pdf_file_path = "data/test_cv.pdf"
model_name="gpt-4"
prompt_template="""Write a concise summary of the following:
    "{data}"
    CONCISE SUMMARY:"""
document_variable_name = "data"
print(stuff_summarise_pdf(pdf_file_path, model_name, prompt_template, document_variable_name))
