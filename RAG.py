from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import faiss
import os

# Load the FLAN-T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
generator_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
chat_history = []
# Preprocess file function (simple text processing)
def preprocess_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text.split("\n")

# Function to convert text to embeddings
# Function to convert text to embeddings
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.encoder(**inputs)  # Use the encoder directly
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling of token embeddings
    return embeddings


# Process the file and convert each paragraph to embedding
def process_and_embed(file_path):
    text_lines = preprocess_file(file_path)
    embeddings = []
    for line in text_lines:
        emb = text_to_embedding(line)
        embeddings.append(emb)
    return np.vstack(embeddings)

# Function to build FAISS index

def build_faiss_index(embeddings, dim=1024):  # Adjust dim based on your embedding model's output
    # Ensure embeddings are of the correct shape and type
    embeddings = np.array(embeddings, dtype=np.float32)  # Convert to float32
    assert embeddings.shape[1] == dim, f"Embedding dimension mismatch. Expected {dim}, but got {embeddings.shape[1]}."

    # Create a FAISS index using L2 distance (Euclidean distance)
    index = faiss.IndexFlatL2(dim)  # L2 distance for similarity search

    # Add embeddings to the index
    index.add(embeddings)  # Add the embeddings to the index
    return index

# Function to retrieve the relevant text indices using FAISS
def retrieve_relevant_text(query, index, top_k=5):
    query_embedding = text_to_embedding(query)
    # Search for the top_k most similar text lines
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]  # Return only the indices (ignoring distances)

# Function to generate the answer from the LLM
def generate_answer(query, relevant_indices, text_lines, top_k=5):
    # Ensure relevant_indices is a list of integers (convert if it's not)
    if isinstance(relevant_indices, np.ndarray):
        relevant_indices = relevant_indices.tolist()

    # Concatenate the top-k relevant texts (assuming relevant_indices is a list of indices)
    context = " ".join([text_lines[i] for i in relevant_indices[:top_k]])  # Top-k context

    # Prepare the input to the generator model (prompt + context)
    prompt = f"Answer the question based on the following context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate the answer using the model
    outputs = generator_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)

    # Decode the generated answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_history.append(f"Question: {query}")
    chat_history.append(f"Answer: {generated_text}")
    return generated_text

# Main RAG pipeline
def rag_pipeline(query, file_path, top_k=5):
    # Preprocess and embed the file
    embeddings = process_and_embed(file_path)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Retrieve relevant passages
    relevant_indices = retrieve_relevant_text(query, index, top_k)

    # Generate the answer using the language model
    text_lines = preprocess_file(file_path)
    generated_answer = generate_answer(query, relevant_indices, text_lines, top_k)

    return generated_answer

# Example query

# query = "What is the impact of artificial intelligence on society?"
# file_path = "/content/newpast.txt"
# answer = rag_pipeline(query, file_path)
# print(answer)

n=0
while n!=1:
  query = input("Enter your question: ")  # Take the user query as input
  if query=='exit':
    break
  file_path = "/content/newpast.txt"
  answer = rag_pipeline(query, file_path)
  print(answer)