from ebooklib import epub

def convert_txt_to_epub(txt_path, epub_path):
    # Read the content from the TXT file
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Create a new EPUB book
    book = epub.EpubBook()

    # Set metadata
    book.set_title('Sample Book Title')
    book.set_language('en')

    # Add author(s)
    book.add_author('Author Name')

    # Create a single chapter
    c1 = epub.EpubHtml(title='Content', file_name='content.xhtml', lang='en')
    c1.content = content

    # Add chapter to the book
    book.add_item(c1)

    # Add navigation files
    book.toc = (epub.Link('content.xhtml', 'Content', 'content'), )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define spine
    book.spine = ['nav', c1]

    # Save EPUB file
    epub.write_epub(epub_path, book, {})
    print(f"EPUB file saved at {epub_path}")

# Example usage
convert_txt_to_epub('canon.txt', 'canon.epub')
