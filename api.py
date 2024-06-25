import os
import json
import boto3
from flask import Flask, request, jsonify, make_response, send_file, render_template, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__,  static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes

# Directory where files are saved
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
s3 = boto3.client('s3', region_name=AWS_REGION)
sqs = boto3.client('sqs', region_name=AWS_REGION)

# Allowed extensions
ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file_path, bucket_name, object_name):
    try:
        print(file_path)
        s3.upload_file(file_path, bucket_name, object_name)
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

# def upload_file_to_s3(file_path, bucket_name, object_name):
#     try:
#         print(file_path)
#         with open(file_path, 'rb') as file:
#             s3.upload_fileobj(file, bucket_name, object_name)
#         return True
#     except Exception as e:
#         print(f"Error uploading file to S3: {e}")
#         return False


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# Flask routes
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate a unique ID for this file
        unique_id = str(uuid.uuid4())
        s3_object_key = f"{filename.rsplit('.', 1)[0]}_{unique_id}.{filename.rsplit('.', 1)[1]}"
        
        # Upload the file to S3
        if upload_file_to_s3(filepath, S3_BUCKET_NAME, s3_object_key):
            # Send a message to SQS with the S3 object reference
            message = {
                's3_bucket': S3_BUCKET_NAME,
                's3_key': s3_object_key,
                'unique_id': unique_id,
                'source_lang': request.form.get('source_lang', 'en'),
                'target_lang': request.form.get('target_lang', 'vi'),
                'recipient_email': request.form.get('recipient_email', '')
            }
            
            sqs.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(message)
            )
            
            return jsonify({"message": "File uploaded successfully and message sent to queue"}), 200
        else:
            return jsonify({"error": "Failed to upload file to S3"}), 500

    return jsonify({"error": "File type not allowed"}), 400

@app.route("/supported_langs", methods=["GET"])
def fetch_supported_langs():
    try:
        supported_languages = [
            {"name": "German", "code": "de"},
            {"name": "English", "code": "en"},
            {"name": "Spanish", "code": "es"},
            {"name": "French", "code": "fr"},
            {"name": "Japanese", "code": "ja"},
            {"name": "Korean", "code": "ko"},
            {"name": "Thai", "code": "th"},
            {"name": "Vietnamese", "code": "vi"},
            {"name": "Chinese", "code": "zh"},
        ]
        return jsonify(supported_languages), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5555)
