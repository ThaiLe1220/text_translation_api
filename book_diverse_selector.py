import os
import random
import nltk
from collections import defaultdict
from tqdm import tqdm
import shutil
import re

# Ensure you have nltk data downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# Paths to the books directories
books_number = 2000
books_dir = "books"
selected_books_dir = f"books_categorized_{books_number}"
normalized_books_dir = f"books_categorized_{books_number}_normalized"


# Define a set of allowed characters (ASCII range for English text, including scientific characters)
allowed_chars = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:'\"!?()[]{}-+*/=<>^%&|~`@#$\\_"
    "π∑√∫µθ∞∆αβγδεζηθικλμνξοπρστυφχψω"
)


# Create the output directories if they don't exist
os.makedirs(selected_books_dir, exist_ok=True)
os.makedirs(normalized_books_dir, exist_ok=True)


# Function to categorize titles
def categorize_titles(titles):
    categories = defaultdict(list)
    for title in tqdm(titles, desc="Categorizing Titles"):
        tokens = nltk.word_tokenize(title)
        tagged = nltk.pos_tag(tokens)
        # Simple categorization based on the most common noun or adjective
        category = "Other"
        for word, tag in tagged:
            if tag in ("NN", "JJ"):
                category = word
                break
        categories[category].append(title)
    return categories


# Function to clean and normalize the text
def clean_text(text):
    # Remove Project Gutenberg headers and footers
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1:
        text = text[start_idx + len(start_marker) :]
    if end_idx != -1:
        text = text[:end_idx]

    # Remove metadata (title, author, etc.)
    text = re.sub(r"Title:.*\n", "", text)
    text = re.sub(r"Author:.*\n", "", text)
    text = re.sub(r"Illustrator:.*\n", "", text)
    text = re.sub(r"Release date:.*\n", "", text)
    text = re.sub(r"Language:.*\n", "", text)
    text = re.sub(r"Original publication:.*\n", "", text)
    text = re.sub(r"Credits:.*\n", "", text)

    # Join lines that are part of the same sentence
    text = re.sub(r"\n(?!\n)", " ", text)  # Replace single newlines with a space
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace

    # Remove non-standard characters
    text = "".join(ch for ch in text if ch in allowed_chars)

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Filter sentences based on length criteria and strip whitespace
    filtered_sentences = [
        sentence.strip() for sentence in sentences if 13 < len(sentence.strip()) < 512
    ]

    return filtered_sentences


# Function to process and save the text files
def process_files(input_dir, output_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    for file in tqdm(files, desc="Processing Files"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()

        sentences = clean_text(text)

        # Save the cleaned text
        with open(os.path.join(output_dir, f"{file}"), "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")


# Function to select 200 books from the categorized titles
def select_books():
    # List all txt files in the directory
    all_books = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Categorize all book titles
    categories = categorize_titles(all_books)

    # Sample with bias to ensure variety with a maximum of 1 book per category and up to 200 categories
    selected_books = []
    category_count = 0

    for category, titles in tqdm(categories.items(), desc="Selecting Books"):
        if category_count >= books_number:
            break
        if len(titles) > 0:
            selected_books.append(random.choice(titles))
            category_count += 1

    # If we have less than {books_number}, fill the remaining with random books
    if len(selected_books) < books_number:
        remaining_books = set(all_books) - set(selected_books)
        selected_books.extend(
            random.sample(remaining_books, books_number - len(selected_books))
        )

    # Save selected books into the output directory
    for book in tqdm(selected_books, desc="Saving Books"):
        shutil.copy(
            os.path.join(books_dir, book), os.path.join(selected_books_dir, book)
        )

    # # Print selected books
    # print("Selected books:", selected_books)


# Function to normalize the selected books
def normalize_books():
    process_files(selected_books_dir, normalized_books_dir)


# Function to count sentences in all normalized books
def count_sentences(directory):
    total_sentences = 0
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    for file in tqdm(files, desc="Counting Sentences"):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            sentences = f.readlines()
            total_sentences += len(sentences)

    return total_sentences


# Main function to run the entire process
def main():
    print(f"Select {books_number} books")
    select_books()

    print("Normalize selected books")
    normalize_books()

    total_sentences = count_sentences(normalized_books_dir)
    print(f"Total number of sentences: {total_sentences}")


if __name__ == "__main__":
    main()
