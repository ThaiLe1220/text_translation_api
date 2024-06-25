from flask import Flask, jsonify, request, send_from_directory, send_file
import pandas as pd

app = Flask(__name__, static_url_path='', static_folder='static')

# Load the book data
book_data = pd.read_csv('book_info_final.csv').fillna('')

# Helper function to get distinct values for filtering
def get_distinct_values(column):
    return book_data[column].dropna().unique().tolist()

@app.route('/api/books', methods=['GET'])
def get_books():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    search_query = request.args.get('search', '')
    
    filters = {}
    for key in request.args.keys():
        if key not in ['page', 'per_page', 'search']:
            filters[key] = request.args.get(key)

    filtered_data = book_data

    if search_query:
        filtered_data = filtered_data[
            filtered_data.apply(lambda row: search_query.lower() in row.astype(str).str.lower().to_string(), axis=1)
        ]

    for column, value in filters.items():
        if value:
            filtered_data = filtered_data[filtered_data[column].str.contains(value, case=False, na=False)]

    start = (page - 1) * per_page
    end = start + per_page

    books_slice = filtered_data[start:end]

    books_json = books_slice.to_dict(orient='records')

    return jsonify(books_json)

@app.route('/api/filters', methods=['GET'])
def get_filters():
    filters = {
        'Author': get_distinct_values('Author'),
        'LoC Class': get_distinct_values('LoC Class'),
        'Subjects': get_distinct_values('Subjects')
    }
    return jsonify(filters)

@app.route('/covers/<path:filename>')
def serve_cover_image(filename):
    return send_from_directory('covers', filename)

@app.route('/')
def index():
    return send_file('static/index.html')

if __name__ == '__main__':
    app.run(debug=True)
