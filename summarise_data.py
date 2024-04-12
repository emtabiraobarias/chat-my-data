from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser
import json

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

import os
from pathlib import Path

class Employment(BaseModel):
    role: str = Field(description="Job Title")
    company: str = Field(description="Company")
    responsibilities: str = Field(description="Summary of responsibilities and achievements during the employment")
    date: str = Field(description="Employment duration, include month and year only")

class Contact(BaseModel):
    address: str = Field(description="Applicant's home address")
    mobile: str = Field(description="Applicant's phone number")
    email: str = Field(description="Applicant's email address")
    references: List[str] = Field(description="List of References provided, if any")

class CV(BaseModel):
    name: str = Field(description="Applicant")
    profile: str = Field(description="Summary Profile")
    skills: List[str] = Field(description="List of Skills and Specialisations")
    employment: List[Employment] = Field(description="Employment History showing the company and the role sorted by latest.")
    trainings: List[str] = Field("List of Trainings and Education taken exclude any dates")
    certification: List[str] = Field(description="List of Certifications exclude the dates")
    contact: Contact = Field("Applicant's contact information")
    link: str = Field(description="Link to document location")

def load_pdf(pdf_file_path="data/test_cv.pdf"):
    print("Loading data...")
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()
    return pages

def define_llm(model_name="gpt-4"):
    print("Defining LLM...")
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    return llm

def summarise_content(pages, prompt, llm, parser):
    print("Summarising content...")
    chain = prompt | llm | parser
    response = chain.invoke({
        "context": pages
    })
    return response

def bulk_summarise(folder_path="data/"):
    file_extension = ('.pdf')
    for files in os.scandir(folder_path):
        if files.path.endswith(file_extension):
            file_path = folder_path + files.name
            json_path = folder_path + 'json/' + Path(file_path).stem + '_data.json'
            if os.path.exists(json_path):
                print("Ignoring " + file_path + " because the summarised json data already exists")
            else: 
                print("Loading document: " + file_path)
                pages = load_pdf(pdf_file_path=file_path)
                llm = define_llm()
                response = summarise_content(pages, prompt, llm, parser)
                print(response)
                with open(json_path, 'w') as f:
                    json.dump(response, f)

##main
#TODO: Save to vector store
#TODO: Cluster common applicants https://github.com/mendableai/QA_clustering/blob/main/notebooks/clustering_approach.ipynb
#TODO: Query with RAG https://python.langchain.com/docs/expression_language/get_started/?ref=gettingstarted.ai + https://python.langchain.com/docs/use_cases/question_answering/
#TODO: Data cleansing: https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625 
parser = JsonOutputParser(pydantic_object=CV)
prompt = PromptTemplate(
    template="""
        You are an expert Recruiter going through an applicant resume.
        Help me create a summary profile of the applicant containing previous job roles, skills and certifications.
        If the information is not present, write UNKNOWN.
        Extract the information as specified.
        {format_instructions}
        {context}
        """,
    description="CV Summary",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
bulk_summarise()
