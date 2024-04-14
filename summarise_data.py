from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser
import json

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

import os
from pathlib import Path
from datetime import datetime

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

def bulk_summarise(folder_path="data/", prompt=prompt, parser=parser):
    file_extension = ('.pdf')
    for files in os.scandir(folder_path):
        if files.path.endswith(file_extension):
            file_path = folder_path + files.name
            json_path = folder_path + 'json/processed_' + Path(file_path).stem + '_data.json'
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

def create_summary_index(folder_path="data/json/"): 
    file_extension = ('.json')
    index = ""
    create_index = 0
    for files in os.scandir(folder_path):
        if files.path.endswith(file_extension) and not files.name.startswith("processed_"):
            try:
                file_path = folder_path + files.name
                json_file = open(file_path)
                json_data = json.load(json_file)
                index = index + json_data["name"] + " is " + json_data["profile"] + "\n\n"
                json_file.close()
                print("Marking '" + file_path + "' as processed")
                processed_file_path = folder_path + "processed_" + files.name
                os.rename(file_path, processed_file_path)
                create_index += 1
            except:
                print("Warning: error in processing '" + files.name + "' moving on with other data files...")
    try:
        if create_index > 0:
            index_filename = folder_path + "index/" + str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "_index.txt"
            print("Creating summary index file in  " + index_filename)
            index_filename = Path(index_filename)
            index_filename.parent.mkdir(exist_ok=True, parents=True)
            with open(index_filename, "x") as index_file:
                index_file.write(index)
    except:
        print("Error in creating index file " + index_filename)

##main
bulk_summarise()
create_summary_index()
