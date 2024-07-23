# How to setup and run the bot

### Installation

```sh
## Clone the repository
git clone <repository_url>

## Create the necessary folders
mkdir db
mkdir models
## Add your model files to the 'models' folder
mkdir docs
----
## Download the model

The model used can be found at https://huggingface.co/MBZUAI/LaMini-T5-738M
Clone the repo, and place the model files in the 'models' folder
Change the path for the model files appropriately in the qa_chatbot.py script.

## Put default documents in place

Make sure the Minesottaplans pdf is present in the 'docs' folder
Change the path for the pdf files appropriately in the qa_chatbot.py and ingest.py scripts.
 
## Run the ingestion script to prepare the data

This script prepares the defalut knowledge base for the query system.
`python ingest.py`

## Start the chatbot application using Streamlit

`streamlit run qa_chatbot.py`

## Note
All imports use the latest possible versions of the concerned libraries, however some methods used in the code may be deprecated, and can be fixed with syntax changes.
