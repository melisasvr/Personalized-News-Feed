# Personalized News Feed
The Personalized News Feed is a Python-based application that recommends news articles based on user interests. Leveraging BERT embeddings from the Hugging Face transformers library, it analyzes article content and matches it to user-provided topics (e.g., "technology, sports, health"). The system boosts recommendations for articles in matching categories and includes a spellchecker to handle typos, ensuring a robust and user-friendly experience.
This project is designed as a proof-of-concept with a static dataset but can be extended to scrape real-time news or load from external sources.

## Features
- Semantic Recommendations: Uses BERT to understand article content and user interests beyond simple keyword matching.
- Category Boosting: Articles matching user-specified categories receive a similarity score boost for better relevance.
- Spellchecking: Corrects typos in user input (e.g., "techonogy" → "technology").
- Neat Output: Displays recommendations in a clean, tabular format with titles, categories, and similarity scores.
- Diverse Dataset: Includes articles across technology, sports, health, politics, business, entertainment, and environment.

## Requirements
- Python: 3.7 or higher
- Libraries:
- pandas
- torch
- transformers
- pyspellchecker

## Installation
1. Clone or Download: Get the project files from the repository or copy the script.
2. Install Dependencies: Run the following command in your terminal:
- pip install pandas torch transformers pyspellchecker
3. Verify Setup: Ensure Python is installed and the script runs without errors.

## Usage
1. Run the Script:
- Open a terminal in the project directory.
- Execute:
- python news_feed.py
2. Enter Interests:
- When prompted, type your interests separated by commas (e.g., "technology, sports, health").
- The script will process your input, correct typos if needed, and display recommendations.
3. View Output:
- Recommended Articles: Top 3 articles most relevant to your interests.
- All Articles: Full list sorted by relevance, including categories and similarity scores.
- Example:
- Computing article embeddings...

```
Personalized News Feed
------------------------------
Enter your interests (e.g., technology, sports, separated by commas): technology, sports, health
Processed interests: technology sports health

Recommended Articles
------------------------------
Sports Teams Compete in Championship          | Sports               | Similarity: 0.614
New AI Technology Improves Healthcare         | Technology, Health   | Similarity: 0.603
Breakthrough in Quantum Computing             | Technology           | Similarity: 0.541

All Articles (Sorted by Relevance)
------------------------------------------------------------
Sports Teams Compete in Championship          | Sports               | Similarity: 0.614
New AI Technology Improves Healthcare         | Technology, Health   | Similarity: 0.603
Breakthrough in Quantum Computing             | Technology           | Similarity: 0.541
[...]
```

## Project Structure
- news_feed.py: The main script contains all the logic, including the dataset, embeddings, and recommendation system.
- Dataset: Static list of 15 articles with titles, content, and categories (embedded in the script).
- Dependencies: External libraries managed via pip.


## How It Works
- Dataset Loading: A predefined set of articles is loaded into a pandas DataFrame.
- BERT Embeddings: BERT converts article content (title + text) and user interests to semantic embeddings.
- Similarity Calculation: Cosine similarity is computed between user and article embeddings.
- Category Boost: Articles matching user interest categories get a 0.1 similarity boost.
- Output: Top 3 recommendations and a fully sorted list are displayed.

## Customization
- Add More Articles: Edit the create_news_dataset() function to include additional titles, content, and categories.
- Adjust Boost: Modify CONFIG["category_boost"] to change how much matching categories influence rankings.
- Change Top N: Update CONFIG["top_n_recommendations"] to show more or fewer top articles.
- Dynamic Data: Replace the static dataset with a CSV loader or web scraper (ensure compliance with terms of use).

## License
- This project is open-source and free to use under the MIT License. 

