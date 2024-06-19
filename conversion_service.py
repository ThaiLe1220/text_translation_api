import os
import time
import json
import boto3
import uuid
from threading import Thread
from nltk.tokenize import sent_tokenize
from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub, TranslatorCT2fromHfHub
from transformers import AutoTokenizer
from dotenv import load_dotenv
from sendmail import send_secure_email  # Ensure this is your function for sending emails

# Load environment variables
load_dotenv()

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
sqs = boto3.client('sqs', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)

# Load the base model names from the configuration file
def load_model_names(file_path):
    model_names = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            model_names[key] = value
    return model_names

weights_relative_path = os.getenv("MODEL_DIR")
model_names = load_model_names("model_names.cfg")
direct_model_mapping = {
    k: f"{weights_relative_path}/ct2fast-{v}" for k, v in model_names.items()
}
supported_langs = ["en", "es", "fr", "de", "zh", "vi", "ko", "th", "ja"]
model_name = "michaelfeil_ct2fast-m2m100_1.2B"
model_dir = os.path.join(weights_relative_path, model_name)
os.makedirs(model_dir, exist_ok=True)

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_1.2B")
model_m2m = MultiLingualTranslatorCT2fromHfHub(
    model_name_or_path="michaelfeil/ct2fast-m2m100_1.2B",
    device="cuda",
    compute_type="int8_float16",
    tokenizer=tokenizer,
)
tokenizer.save_pretrained(model_dir)
if hasattr(model_m2m, "save_pretrained"):
    model_m2m.save_pretrained(model_dir)
else:
    print("The model does not support the 'save_pretrained' method. Please implement a custom saving mechanism.")

print(f"Model and tokenizer have been saved to '{model_dir}'")

# Function to split text into chunks
def split_text(text, max_length):
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]
    return text_chunks

# Translation functions
def translate_text(sentences, src_lang, tgt_lang):
    outputs = model_m2m.generate([sentences], src_lang=[src_lang], tgt_lang=[tgt_lang])
    return outputs[0]

def translate_with_timing(text, source_lang, target_lang):
    def perform_translation(text, model_dir):
        model = TranslatorCT2fromHfHub(
            model_name_or_path=model_dir,
            device="cuda",
            compute_type="int8_float16",
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
        )
        translated_text = model.generate(text=text)
        return translated_text

    if f"{source_lang}-{target_lang}" in ["en-ko", "en-th", "en-ja"]:
        translated_text = translate_text(text, source_lang, target_lang)
        return translated_text

    if f"{source_lang}-{target_lang}" in direct_model_mapping:
        translated_text = perform_translation(
            text, direct_model_mapping[f"{source_lang}-{target_lang}"]
        )
    elif source_lang in supported_langs and target_lang in supported_langs:
        intermediate_text = perform_translation(
            text, direct_model_mapping[f"{source_lang}-en"]
        )
        final_text = perform_translation(
            intermediate_text, direct_model_mapping[f"en-{target_lang}"]
        )
        translated_text = final_text
    else:
        translated_text = translate_text(text, source_lang, target_lang)
    return translated_text



def process_file(s3_bucket, s3_key, source_lang, target_lang, unique_id,recipient_email):
    # Download the file from S3
    local_file_path = f"/tmp/{s3_key.split('/')[-1]}"
    s3.download_file(s3_bucket, s3_key, local_file_path)

    # Read the file content
    with open(local_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Translate each non-empty line, preserving line breaks
    translated_lines = []
    for line in lines:
        if not line.strip():
            # Preserve empty lines
            translated_lines.append("\n\n")
        else:
            # Translate non-empty lines
            if len(tokenizer.tokenize(line)) > 512:
                line_chunks = split_text(line, 512)
                translated_chunks = [translate_with_timing(chunk, source_lang, target_lang) for chunk in line_chunks]
                translated_lines.extend(translated_chunks)
            else:
                translated_lines.append(" " +translate_with_timing(line, source_lang, target_lang))
    
    # Combine translated lines into content
    translated_content = "".join(translated_lines)

    # Save the translated content to a new file
    translated_file_name = f"{s3_key.rsplit('.', 1)[0]}_{unique_id}_translated.txt"
    translated_file_path = f"/tmp/{translated_file_name}"
    with open(translated_file_path, 'w', encoding='utf-8') as file:
        file.write(translated_content)

    # Upload the translated file back to S3
    s3.upload_file(translated_file_path, s3_bucket, translated_file_name)

    presigned_url = upload_file_to_s3(translated_file_path, s3_bucket, translated_file_name)

    # Send email notification with the download link
    email_subject = "Your book is ready!"
    email_body = f"Your processed book is ready. You can download it from: {presigned_url} within 7 days"
    send_secure_email(email_subject, email_body, recipient_email, EMAIL, EMAIL_PASSWORD)
    print(f"Email sent to {recipient_email}")


def upload_file_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        # Generate a presigned URL for the uploaded file
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                         Params={'Bucket': bucket_name,
                                                                 'Key': object_name},
                                                         ExpiresIn=3600*24*7) # URL expires in 7 days
        return presigned_url
    except Exception:
        print("Credentials not available")
        return None

def process_sqs_message():
    while True:
        try:
            time.sleep(2)
            message_body = read_message_from_sqs()
            if message_body:
                s3_bucket = message_body['s3_bucket']
                s3_key = message_body['s3_key']
                source_lang = message_body['source_lang']
                target_lang = message_body['target_lang']
                unique_id = message_body['unique_id']
                recipient_email = message_body['recipient_email']
                process_file(s3_bucket, s3_key, source_lang, target_lang, unique_id, recipient_email)
        except Exception as e:
            print(f"An error occurred: {e}")

# Function to read message from SQS
def read_message_from_sqs():
    response = sqs.receive_message(
        QueueUrl=SQS_QUEUE_URL,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=30,
        WaitTimeSeconds=0
    )

    if 'Messages' in response:
        for message in response['Messages']:
            body = message['Body']
            receipt_handle = message['ReceiptHandle']
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            return json.loads(body)
    return None

# Start the conversion service
if __name__ == "__main__":
    process_sqs_message()
