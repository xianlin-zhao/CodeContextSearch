import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
import torch

# Load the model, optionally in float16 precision for faster inference
model = SentenceTransformer(
    "BAAI/bge-code-v1",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16},
)

# Prepare a prompt given an instruction
instruction = 'Given a question in text, retrieve SQL queries that are appropriate responses to the question.'
prompt = f'<instruct>{instruction}\n<query>'
# Prepare queries and documents
queries = [
    "Delete the record with ID 4 from the 'Staff' table.", 
    'Delete all records in the "Livestock" table where age is greater than 5'
]
documents = [
    "DELETE FROM Staff WHERE StaffID = 4;",
    "DELETE FROM Livestock WHERE age > 5;"
]

# Compute the query and document embeddings
query_embeddings = model.encode(queries, prompt=prompt)
document_embeddings = model.encode(documents)

# Compute the cosine similarity between the query and document embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)