# Severity-of-Toxic-Comments-MLOps


##  Methodology
-Data Analysis
-Data preprocessing
-Text Normalisation
-Lemmatization
-Stop-words Removal
-Tokenization
-Embedding words into vectors using FastText
-Trained Model using LSTM, LSTM-CNN
-Model Evaluation
-Achieved the best accuracy using LSTM

## Workflow - deployment
-Saved the tokenizer and the DL model.
-Developed backend and frontend using Python and Streamlit library, deployed the web app on localhost.
-Containerized the software using Docker and tested it on localhost.
-Hosted EC2 instance on AWS and installed Docker. I connected the EC2 to GitHub using GitHub actions for the CI/CD pipeline.
-Used Elastic Container Registry (ECR) to store the Docker image
-Acurracy : Train Data - 99.08% Test Data - 99.32%
