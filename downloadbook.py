import os
import requests
import csv
import time

# Ensure the base directory exists for downloaded books
download_dir = "books"
os.makedirs(download_dir, exist_ok=True)

# Error log file
error_log_file = "error_books.csv"
last_processed_file = "last_processed_row.txt"


# Function to log errors
def log_error(book_title, book_number, error_message):
    with open(error_log_file, mode="a", encoding="utf-8", newline="") as error_file:
        error_writer = csv.writer(error_file)
        error_writer.writerow([book_title, book_number, error_message])


# Function to read the error log
def read_error_log():
    error_books = set()
    if os.path.exists(error_log_file):
        with open(error_log_file, mode="r", encoding="utf-8") as error_file:
            error_reader = csv.reader(error_file)
            for row in error_reader:
                if len(row) >= 2:
                    error_books.add((row[0], row[1]))
    return error_books


# Function to create a safe file path
def create_safe_file_path(book_title, book_number):
    safe_book_title = book_title.replace("/", "-").replace('"', "")
    return os.path.join(download_dir, f"{safe_book_title}_{book_number}.txt")


# Function to read the last processed row index
def read_last_processed_row():
    if os.path.exists(last_processed_file):
        with open(last_processed_file, mode="r", encoding="utf-8") as file:
            return int(file.read().strip())
    return 0


# Function to update the last processed row index
def update_last_processed_row(row_index):
    with open(last_processed_file, mode="w", encoding="utf-8") as file:
        file.write(str(row_index))


# Read error log
error_books = read_error_log()
last_processed_row = read_last_processed_row()

# Read the CSV file
processed_count = 0
with open("books.csv", mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row_index, row in enumerate(csv_reader, start=1):
        if row_index <= last_processed_row:
            continue

        book_title = row["Book Title"].replace('"', "")
        book_number = row["Book Number"]
        book_url = (
            f"https://www.gutenberg.org/cache/epub/{book_number}/pg{book_number}.txt"
        )
        file_path = create_safe_file_path(book_title, book_number)

        # Ensure the directory for the file path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if the book has already been downloaded or is in the error log
        if os.path.exists(file_path):
            continue
        if (book_title, book_number) in error_books:
            continue

        success = False
        attempts = 0
        max_attempts = 1

        while not success and attempts < max_attempts:
            try:
                response = requests.get(book_url, timeout=10)
                response.raise_for_status()
                success = True
            except requests.exceptions.RequestException as e:
                attempts += 1
                time.sleep(0.5)

        if success:
            with open(file_path, "w", encoding="utf-8") as book_file:
                book_file.write(response.text)
            processed_count += 1
            update_last_processed_row(row_index)
        else:
            error_message = f"Failed to download after {max_attempts} attempts"
            log_error(book_title, book_number, error_message)

print(f"Processing completed. Total books processed: {processed_count}.")
