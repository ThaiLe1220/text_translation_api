import csv
import os
import requests
from bs4 import BeautifulSoup

# Function to fetch the book information from the webpage
def fetch_book_info(book_id):
    try:
        url = f"https://www.gutenberg.org/ebooks/{book_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Initialize the book info dictionary with predefined field names
        book_info = {
            "Book ID": book_id,
            "Author": "",
            "Illustrator": "",
            "Title": "",
            "Original Publication": "",
            "Credits": "",
            "Language": "",
            "LoC Class": "",
            "Subjects": [],
            "Category": "",
            "EBook-No.": "",
            "Release Date": "",
            "Copyright Status": "",
            "Downloads": "",
            "Cover Image URL": f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.cover.medium.jpg"
        }

        # Find the table containing the book information
        table = soup.find('table', class_='bibrec')
        rows = table.find_all('tr')

        for row in rows:
            th = row.find('th')
            td = row.find('td')
            
            if th and td:
                th_text = th.get_text(strip=True)
                td_text = td.get_text(strip=True)
                
                if th_text == "Author":
                    book_info["Author"] = td_text
                elif th_text == "Illustrator":
                    book_info["Illustrator"] = td_text
                elif th_text == "Title":
                    book_info["Title"] = td_text
                elif th_text == "Original Publication":
                    book_info["Original Publication"] = td_text
                elif th_text == "Credits":
                    book_info["Credits"] = td_text
                elif th_text == "Language":
                    book_info["Language"] = td_text
                elif th_text == "LoC Class":
                    book_info["LoC Class"] = td_text
                elif th_text == "Subject":
                    book_info["Subjects"].append(td_text)
                elif th_text == "Category":
                    book_info["Category"] = td_text
                elif th_text == "EBook-No.":
                    book_info["EBook-No."] = td_text
                elif th_text == "Release Date":
                    book_info["Release Date"] = td_text
                elif th_text == "Copyright Status":
                    book_info["Copyright Status"] = td_text
                elif th_text == "Downloads":
                    book_info["Downloads"] = td_text
        
        # Join subjects into a single string
        book_info["Subjects"] = "; ".join(book_info["Subjects"])
        
        return book_info
    except Exception as e:
        print(e)
        return ""


# Function to download the cover image
def download_cover_image(book_id):
    cover_image_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.cover.medium.jpg"
    response = requests.get(cover_image_url)
    
    if response.status_code == 200:
        with open(f'covers/{book_id}.jpg', 'wb') as f:
            f.write(response.content)

# Create covers directory if it doesn't exist
if not os.path.exists('covers'):
    os.makedirs('covers')

# Read the list of book IDs from books.csv
with open('books.csv', 'r') as f:
    reader = csv.reader(f)
    book_list = [(row[0], row[1]) for row in reader]

# Open the output CSV file for writing
with open('book_info.csv', 'w', newline='') as csvfile:
    fieldnames = [
        "Book ID", "Author", "Illustrator", "Title", "Original Publication", "Credits", 
        "Language", "LoC Class", "Subjects", "Category", "EBook-No.", 
        "Release Date", "Copyright Status", "Downloads", "Cover Image URL"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each book ID, fetch the book information, and write to the CSV file
    for book_title, book_id in book_list:
        try:
            print("Scrapping on "+book_title)
            book_info = fetch_book_info(book_id)
            writer.writerow(book_info)
            download_cover_image(book_id)
        except Exception as e:
            print(e)

print("Book information has been successfully written to book_info.csv.")
