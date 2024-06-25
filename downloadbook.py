import os
import requests
import csv

# Ensure the directory exists for downloaded books
download_dir = 'books'
os.makedirs(download_dir, exist_ok=True)

# Read the CSV file
with open('books.csv', mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        book_title = row['Book Title']
        book_number = row['Book Number']
        book_url = f'https://www.gutenberg.org/cache/epub/{book_number}/pg{book_number}.txt'
        
        response = requests.get(book_url)
        
        if response.status_code == 200:
            file_path = os.path.join(download_dir, f'{book_title}_{book_number}.txt')
            with open(file_path, 'w', encoding='utf-8') as book_file:
                book_file.write(response.text)
            print(f'Downloaded: {book_title} (Book Number: {book_number})')
        else:
            print(f'Failed to download: {book_title} (Book Number: {book_number})')
