from sentence_transformers import SentenceTransformer
import torch

# # Load model
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# # Define two example sentences
# sentence1 = "The quick brown fox jumps over the lazy dog"
# sentence2 = "A quick brown dog jumps over the lazy fox"


# # Encode sentences and compute similarity score
# embeddings1 = model.encode([sentence1], convert_to_tensor=True)
# embeddings2 = model.encode([sentence2], convert_to_tensor=True)
# cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)


# # Print similarity score
# print(f"Similarity score: {cosine_similarities.item()}")







# Define a function to compute similarity scores
def compute_similarity_score(source_sentence, compare_sentences):
    # Encode the source sentence
    source_embedding = model.encode(source_sentence, convert_to_tensor=True)

    # Encode the compare sentences
    compare_embeddings = model.encode(compare_sentences, convert_to_tensor=True)

    # Compute cosine similarity scores
    cosine_similarities = torch.nn.functional.cosine_similarity(source_embedding, compare_embeddings)

    # Return the similarity scores
    return cosine_similarities.tolist()


# Import necessary libraries
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch

# Load the pre-trained sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



# Set page configuration
st.set_page_config(page_title="Sentence Similarity Score", page_icon=":guardsman:", layout="wide")

# Set sidebar
st.sidebar.title("Sentence Similarity Score")

# Add input fields to the sidebar
with st.sidebar:
    source_sentence = st.text_input(label="Enter the source sentence:")

    compare_sentences = []
    for i in range(3):
        compare_sentence = st.text_input(label=f"Enter sentence {i+1} to compare (optional):", key=f"compare_sentence_{i}")
        if compare_sentence:
            compare_sentences.append(compare_sentence)
            
            
# Compute similarity scores
if st.button("Compute Similarity Scores"):
    if not source_sentence:
        st.error("Please enter a source sentence.")
    else:
        scores = compute_similarity_score(source_sentence, compare_sentences)

        # Display results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Score")
            st.write("---")
            for i, score in enumerate(scores):
                st.write(f"{i+1}. {score:.2f}")
        with col2:
            st.write("Sentence")
            st.write("---")
            st.write("Source Sentence")
            for i, compare_sentence in enumerate(compare_sentences):
                st.write(f"Compare Sentence {i+1}")

# Add line bar to the page
st.markdown(
    """<style>
        .css-hi6a2p{border-bottom: 2px solid #6EB3D0;}
    </style>""",
    unsafe_allow_html=True,
)