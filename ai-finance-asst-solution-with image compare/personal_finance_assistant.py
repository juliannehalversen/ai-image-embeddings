import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel, pipeline, AutoImageProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import certifi
print("Certifi path:", certifi.where())

# --- Part 1: Tokenizing and Embedding Financial Data ---
@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


def get_embeddings(descriptions, tokenizer, model):
    tokenized_descriptions = [
        tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
        for desc in descriptions
    ]
    embeddings = []
    for tokenized in tokenized_descriptions:
        with torch.no_grad():
            outputs = model(**tokenized)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings


# --- Part 2: Handling Chatbot and Text Completion ---
@st.cache_resource
def load_chatbot_model():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    return pipeline("text-generation", model="gpt2", device=device)


def financial_chatbot(query):
    prompt = f"As a financial assistant, please respond to this query: {query}"
    response = chatbot_model(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]


# --- Part 3: Enabling Search Using LLMs ---
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Load all models and tokenizers
tokenizer, bert_model = load_tokenizer_and_model()
chatbot_model = load_chatbot_model()
sentence_model = load_sentence_transformer()

# Sidebar for navigation
st.sidebar.title("Personal Finance Assistant")
option = st.sidebar.selectbox(
    "Choose a function:", ["Embedding Visualization", "Chatbot", "Transaction Search", "Compare Images"]
)

# Section 1: Embedding Visualization
if option == "Embedding Visualization":
    st.title("Embedding Visualization")
    data = {
        "Description": [
            "Grocery Store",
            "Rent Payment",
            "Coffee Shop",
            "Electricity Bill",
            "Restaurant",
        ],
        "Amount": [150.25, 1200.00, 4.50, 60.75, 35.40],
    }
    df = pd.DataFrame(data)

    embeddings = get_embeddings(df["Description"], tokenizer, bert_model)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for i, desc in enumerate(df["Description"]):
        plt.annotate(desc, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    st.pyplot(plt)

# Section 2: Chatbot
elif option == "Chatbot":
    st.title("Financial Chatbot")
    user_query = st.text_input("Enter your query:")
    if user_query:
        response = financial_chatbot(user_query)
        st.write(f"Chatbot: {response}")

# Section 3: Transaction Search
elif option == "Transaction Search":
    st.title("Transaction Search")
    transactions = [
        "Grocery Store $150.25",
        "Rent Payment $1200.00",
        "Coffee Shop $4.50",
        "Electricity Bill $60.75",
        "Restaurant $35.40",
    ]
    transaction_embeddings = sentence_model.encode(transactions)

    query = st.text_input("Enter your search query (e.g., groceries, rent, coffee):")

    if query:
        query_embedding = sentence_model.encode(query)
        similarities = util.pytorch_cos_sim(
            query_embedding, transaction_embeddings
        ).numpy()[0]
        best_match_idx = similarities.argmax()
        st.write(f"Best matching transaction: {transactions[best_match_idx]}")
        st.write(f"Similarity score: {similarities[best_match_idx]:.4f}")

# Section 4: Compare Images
elif option == "Compare Images":
    # Load the model and processor
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Function to get embeddings from an image
    def get_image_embeddings(image):
        # Preprocess the image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Extract image embeddings using the model
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        
        return embeddings

    # Streamlit UI for uploading two images
    st.title("Image Comparison with Embeddings")

    option = st.selectbox("Select an option", ["Compare Images"])
    if option == "Compare Images":
        uploaded_file_1 = st.file_uploader("Upload the first image (PNG/JPG)", type=["png", "jpg"], key="1")
        uploaded_file_2 = st.file_uploader("Upload the second image (PNG/JPG)", type=["png", "jpg"], key="2")

        if uploaded_file_1 and uploaded_file_2:
            # Open the images using PIL
            image_1 = Image.open(uploaded_file_1)
            image_2 = Image.open(uploaded_file_2)

            # Display the uploaded images
            st.image([image_1, image_2], caption=["First Image", "Second Image"], use_column_width=True)

            # Get the embeddings for both images
            embeddings_1 = get_image_embeddings(image_1)
            embeddings_2 = get_image_embeddings(image_2)

            # Calculate the similarity between the two embeddings using cosine similarity
            similarity = torch.nn.functional.cosine_similarity(embeddings_1, embeddings_2)
            st.write(f"Similarity score: {similarity.item():.4f}")

            # Display embeddings
            st.write("Embeddings for Image 1:", embeddings_1)
            st.write("Embeddings for Image 2:", embeddings_2)