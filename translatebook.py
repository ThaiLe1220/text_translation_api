import os
import subprocess

# Function to translate a book using the curl command
def translate_book(file_path, source_lang, target_lang, recipient_email):
    try:
        curl_command = [
            'curl', '-X', 'POST', 'https://api.booktranslate.uk/upload',
            '-F', f'file=@{file_path}',
            '-F', f'source_lang={source_lang}',
            '-F', f'target_lang={target_lang}',
            '-F', f'recipient_email={recipient_email}'
        ]
        subprocess.run(curl_command, check=True)
        print(f'Translated: {file_path}')
    except subprocess.CalledProcessError as e:
        print(f'Failed to translate: {file_path}. Error: {e}')

# Directory where books are downloaded
download_dir = 'books'

# Iterate over each downloaded book and translate it
for book_file in os.listdir(download_dir):
    if book_file.endswith('.txt'):
        file_path = os.path.join(download_dir, book_file)
        translate_book(file_path, 'en', 'vi', 'thanhnguyenaws2021@gmail.com')
