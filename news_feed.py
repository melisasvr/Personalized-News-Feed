import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import warnings
import os
from spellchecker import SpellChecker

# Configuration settings
CONFIG = {
    "max_articles": 15,          # Maximum articles in dataset
    "top_n_recommendations": 3,  # Number of top recommendations
    "category_boost": 0.1,       # Boost for articles matching user categories
    "similarity_threshold": 0.4  # Minimum similarity for display (optional)
}

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Initialize spellchecker
spell = SpellChecker()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Step 1: Define a larger dataset with diverse categories
def create_news_dataset():
    data = {
        "title": [
            "New AI Technology Improves Healthcare",
            "Sports Teams Compete in Championship",
            "Climate Change Impacts Coastal Cities",
            "Breakthrough in Quantum Computing",
            "Football Season Kicks Off with Surprises",
            "New Tax Policy Sparks Debate in Congress",
            "Stock Market Hits Record High",
            "Blockbuster Movie Breaks Box Office Records",
            "Study Links Diet to Heart Disease Prevention",
            "Global Leaders Meet to Discuss Trade Deals",
            "New Streaming Service Launches with Big Stars",
            "Air Pollution Levels Reach Critical Highs",
            "Tech Giants Face Antitrust Investigations",
            "Pop Star Announces World Tour Dates",
            "Small Businesses Thrive in New Economy"
        ],
        "content": [
            "AI systems are revolutionizing healthcare with new tools and innovations.",
            "Top teams battle it out in an exciting sports season full of surprises.",
            "Rising sea levels threaten urban areas globally due to climate change.",
            "Scientists unveil a new quantum computing method that promises speed.",
            "The football season starts with unexpected wins and dramatic matches.",
            "Lawmakers argue over a new tax policy affecting millions of citizens.",
            "Investors cheer as the stock market reaches an all-time high this week.",
            "A new blockbuster film shatters expectations with massive ticket sales.",
            "Research shows a balanced diet can reduce heart disease risk significantly.",
            "World leaders negotiate trade agreements to boost economic growth.",
            "A streaming platform debuts with exclusive shows featuring top celebrities.",
            "Cities report dangerous air quality as pollution levels spike this month.",
            "Regulators probe major tech companies for monopolistic practices.",
            "A famous pop star reveals plans for a global concert tour next year.",
            "Local entrepreneurs report record profits thanks to recent economic shifts."
        ],
        "category": [
            "Technology, Health", "Sports", "Environment", "Technology", "Sports",
            "Politics", "Business", "Entertainment", "Health", "Politics, Business",
            "Entertainment", "Environment", "Technology, Business", "Entertainment",
            "Business"
        ]
    }
    return pd.DataFrame(data)

# Step 2: Get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Step 3: Calculate cosine similarity
def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = (sum(a * a for a in vec1)) ** 0.5
    norm2 = (sum(b * b for b in vec2)) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

# Step 4: Process user input with spellchecking
def process_user_input():
    raw_input = input("Enter your interests (e.g., technology, sports, separated by commas): ").strip().lower()
    interests = [word.strip() for word in raw_input.split(",")]
    corrected_interests = []
    for interest in interests:
        corrected = spell.correction(interest)
        if corrected != interest:
            print(f"Did you mean '{corrected}' instead of '{interest}'? Using corrected version.")
        corrected_interests.append(corrected)
    return " ".join(corrected_interests), corrected_interests

# Step 5: Adjust similarity with category boost
def adjust_similarity(df, user_categories):
    df["adjusted_similarity"] = df["similarity"]
    for idx, row in df.iterrows():
        article_cats = [cat.strip().lower() for cat in row["category"].split(",")]
        for user_cat in user_categories:
            if user_cat in article_cats:
                df.at[idx, "adjusted_similarity"] += CONFIG["category_boost"]
                break
    return df

# Step 6: Main function
def run_news_feed():
    # Load dataset
    df = create_news_dataset()
    df["combined_text"] = df["title"] + " " + df["content"]
    
    # Compute embeddings
    print("Computing article embeddings...")
    df["embedding"] = df["combined_text"].apply(get_bert_embedding)
    
    # Get user interests
    print("\nPersonalized News Feed")
    print("-" * 30)
    user_interests, user_categories = process_user_input()
    print(f"Processed interests: {user_interests}")
    user_embedding = get_bert_embedding(user_interests)
    
    # Calculate similarity
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity_manual(user_embedding, x))
    
    # Adjust similarity with category boost
    df = adjust_similarity(df, user_categories)
    
    # Sort and select top articles
    top_articles = df.sort_values(by="adjusted_similarity", ascending=False).head(CONFIG["top_n_recommendations"])
    all_articles = df.sort_values(by="adjusted_similarity", ascending=False)
    
    # Display results
    print("\nRecommended Articles")
    print("-" * 30)
    for idx, row in top_articles.iterrows():
        print(f"{row['title']:<45} | {row['category']:<20} | Similarity: {row['adjusted_similarity']:.3f}")
    
    print("\nAll Articles (Sorted by Relevance)")
    print("-" * 60)
    for idx, row in all_articles.iterrows():
        print(f"{row['title']:<45} | {row['category']:<20} | Similarity: {row['adjusted_similarity']:.3f}")

# Run the program
if __name__ == "__main__":
    run_news_feed()