# DC Regulations Website Data Download and Parsing

The Washington DC Regulations website at `https://www.dcregs.dc.gov` is a little outdated and hard to
navigate. The search is a bit limited and the actual regulations are sequestered in individual legacy Microsoft Word `.doc` files.


# Notebooks

## 1. Data Download

The first notebook `0.1_data_download.ipynb` downloads all individual Sections for each chapter of Title 14 (Housing).

I did an initial run of the notebook to get the docs in `./data/housing/pdf` on Feb 8th, 2024.

## 2. Text extraction


The 2nd notebook `0.2_text_extract.ipynb` uses PyMuPDF to extract the text from each PDF document.


## 3. Langchain RAG Q&A over Documents

The 3rd notebook `0.3_langchain.ipynb` builds a basic langchain RAG system over the PDF documents.
Some example queries, their answers, and the source docs identifies are listed at the bottom of the notebook for reference.
