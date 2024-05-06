import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.cluster import KMeans

# pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Read text entries from the file and add them to the text_data list
text_data = []
with open('./training_text_data.txt', 'r') as file:
    lines = file.readlines()
    text_data.extend([line.strip() for line in lines])

# Tokenize and encode text data
encoded_data = [tokenizer.encode(text, return_tensors='pt') for text in text_data]

# Generate embeddings using GPT-2
embeddings = []
with torch.no_grad():
    for data in encoded_data:
        outputs = model(data)
        last_hidden_states = outputs.last_hidden_state
        pooled_embedding = last_hidden_states.mean(dim=1).squeeze().numpy()
        embeddings.append(pooled_embedding)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# faiss for approximate nearest neighbor search
index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance index
index.add(embeddings_np)  # Add embeddings to the index

# Get input text from user
input_text = input("Enter your text question: ")

# Tokenize and encode input text
input_data = tokenizer.encode(input_text, return_tensors='pt')

# Generate embedding for input text
with torch.no_grad():
    inputs = model(input_data)
    input_embedding = inputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Query for nearest neighbors
k = 5  # Number of nearest neighbors
distances, indices = index.search(input_embedding.reshape(1, -1).astype(np.float32), k)

# Retrieve nearest neighbor embeddings
nearest_neighbor_embeddings = embeddings_np[indices[0]]

# Perform k-means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(nearest_neighbor_embeddings)

# Get cluster centroids
cluster_centroids = kmeans.cluster_centers_

# Generate new text from cluster centroids
new_text = []
with torch.no_grad():
    for centroid in cluster_centroids:
        centroid_tensor = torch.tensor([centroid]).float()  # Convert centroid to tensor
        generated_ids = model.generate(input_ids=None, attention_mask=None, encoder_outputs=centroid_tensor)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text.append(generated_text)

# Print generated text
print("Generated Text:")
for i, text in enumerate(new_text):
    print(f"Cluster {i+1}: {text}")
