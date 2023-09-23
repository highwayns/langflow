from langflow.custom.PyPDFLoaderWithOCR import PyPDFLoaderWithOCR
from langchain.schema import Document

pdf_loader = PyPDFLoaderWithOCR('./tests/data/img20230920_17291386.pdf','')
documents = pdf_loader.load()
for doc in documents :
    print(doc.page_content) 