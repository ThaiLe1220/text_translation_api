"""
This script processes text files (books) for translation from one language to another using the
CTranslate2 model from Hugging Face. It handles large files by logging their details in a CSV file
and processes smaller files by splitting text into manageable chunks, translating them, and saving
the translated text to a new file.
"""

import os
import shutil
import concurrent.futures
import csv
from hf_hub_ctranslate2 import (
    MultiLingualTranslatorCT2fromHfHub,
    TranslatorCT2fromHfHub,
)
from transformers import AutoTokenizer
from dotenv import load_dotenv
import re
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()


def load_model_names(file_path):
    """
    Load model names from a configuration file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary mapping model keys to model names.
    """
    model_names = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            model_names[key] = value
    return model_names


# Load model names and initialize model paths
weights_relative_path = os.getenv("MODEL_DIR")
model_names = load_model_names("model_names.cfg")
direct_model_mapping = {
    k: f"{weights_relative_path}/ct2fast-{v}" for k, v in model_names.items()
}
supported_langs = ["en", "vi"]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("weights/ct2fast-mix-en-vi-4m")


def split_text(text, max_length):
    """
    Split a large text into smaller chunks based on sentence-ending punctuation.

    Args:
        text (str): The input text to split.
        max_length (int): Maximum length of each chunk.

    Returns:
        list: List of text chunks.
    """
    sentences = re.split(r'(?<=\.|,|;|"|!|\?)\s+', text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
            current_length += sentence_length + 1

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def translate_with_timing(text_chunk, source_lang, target_lang):
    """
    Translate a text chunk using a specific translation model and measure the timing.

    Args:
        text_chunk (str): The text chunk to translate.
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        str: Translated text.
    """

    def perform_translation(text_chunk, model_dir):
        model = TranslatorCT2fromHfHub(
            model_name_or_path=model_dir,
            device="cuda",
            compute_type="int8_float16",
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
        )
        translated_text = model.generate(text=text_chunk)
        return translated_text

    translated_text = perform_translation(
        text_chunk, direct_model_mapping[f"{source_lang}-{target_lang}"]
    )

    return translated_text


def remove_line(file_path):
    """
    Remove unnecessary line breaks from a text file.

    Args:
        file_path (str): Path to the input text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    processed_lines = []
    for i, line in enumerate(lines):
        if line.strip() == "":
            processed_lines.append(line)
        else:
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                processed_lines.append(line)
            else:
                processed_lines.append(line.rstrip() + " ")

    processed_content = "".join(processed_lines)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(processed_content)


def process_chunk(index, chunk, source_lang, target_lang):
    """
    Translate a single chunk of text.

    Args:
        index (int): Index of the chunk.
        chunk (str): Text chunk to translate.
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        tuple: Index and translated chunk.
    """
    translated_chunk = translate_with_timing(chunk, source_lang, target_lang)
    return index, translated_chunk


def process_file(local_file_path, source_lang, target_lang):
    """
    Process a text file by removing line breaks, splitting text into chunks, translating chunks,
    and writing the translated text to a new file.

    Args:
        local_file_path (str): Path to the input text file.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
    """
    start_time = time.time()

    # Remove unnecessary line breaks from the file
    remove_line(local_file_path)

    # Read the file content
    with open(local_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    read_time = time.time()

    translated_lines = []
    total_chunks = 0
    for line in tqdm(lines, desc="Translating lines"):
        if not line.strip():
            translated_lines.append("\n")
        else:
            # Split line into chunks for translation
            chunks = split_text(line, 512)
            total_chunks += len(chunks)
            max_threads = 32
            translated_chunks = [None] * len(chunks)

            # Translate chunks concurrently
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_threads
            ) as executor:
                futures = [
                    executor.submit(
                        process_chunk, index, chunk, source_lang, target_lang
                    )
                    for index, chunk in enumerate(chunks)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        index, translated_chunk = future.result()
                        translated_chunks[index] = translated_chunk
                    except Exception as e:
                        print(f"Error processing chunk: {e}")

            translated_lines.append(" ".join(filter(None, translated_chunks)) + "\n")

    translate_time = time.time()
    print(f"Time to translate all chunks: {translate_time - read_time:.2f} seconds")

    translated_content = "".join(translated_lines)

    # Write the translated content to a new file
    output_dir = "books_translation"
    os.makedirs(output_dir, exist_ok=True)
    translated_file_name = (
        f"{os.path.basename(local_file_path).rsplit('.', 1)[0]}_translated.txt"
    )
    translated_file_path = os.path.join(output_dir, translated_file_name)
    with open(translated_file_path, "w", encoding="utf-8") as file:
        file.write(translated_content)

    write_time = time.time()
    print(f"Time to write file: {write_time - translate_time:.2f} seconds")
    print(f"Total time for processing file: {write_time - start_time:.2f} seconds")

    print(f"Translated file saved to '{translated_file_path}'")


def process_local_file(file_name, source_lang, target_lang, large_files_writer):
    """
    Process a single text file. If the file has more than 15,000 lines, log it to a CSV file.
    Otherwise, process it for translation.

    Args:
        file_name (str): Name of the text file.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        large_files_writer (csv.writer): CSV writer to log large files.
    """
    local_file_path = os.path.join("books", file_name)
    translated_file_name = (
        f"{os.path.basename(local_file_path).rsplit('.', 1)[0]}_translated.txt"
    )
    translated_file_path = os.path.join("books_translation", translated_file_name)

    # Skip the file if it has already been translated
    if any(
        translated_file_name[:30] in f[:30] for f in os.listdir("books_translation")
    ):
        print(f"Skipping already translated file: {translated_file_name}")
        return

    # Check the line count of the file
    with open(local_file_path, "r", encoding="utf-8") as file:
        line_count = sum(1 for _ in file)

    # Log large files to CSV
    if line_count > 15000:
        large_files_writer.writerow([file_name, line_count])
        print(f"Logged large file to CSV: {file_name}")
        return

    print(f"[WORKFLOW START] Processing file {local_file_path}")

    start_time = time.time()
    process_file(local_file_path, source_lang, target_lang)
    end_time = time.time()

    print(
        f"[WORKFLOW END] Processing time for {file_name}: {end_time - start_time} seconds"
    )


def process_local_books():
    """
    Process all text files in the 'books' directory. Log large files with more than 15,000 lines
    to a CSV file.
    """
    books_dir = "books"
    source_lang = "en"
    target_lang = "vi"
    files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    files.sort()

    with open("books_15k_plus.csv", mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        for file_name in files:
            process_local_file(file_name, source_lang, target_lang, csv_writer)


if __name__ == "__main__":
    process_local_books()
