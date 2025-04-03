import re
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Function to extract important dates from email body
def extract_dates(text):
    date_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for date in matches:
            try:
                dt = datetime.strptime(date, "%B %d")
                dates.append((dt, date))
            except ValueError:
                continue

    dates.sort()
    return [date[1] for date in dates]

# Function to detect the tone of the email
def detect_tone(text):
    result = sentiment_analyzer(text[:512])
    sentiment = result[0]['label']
    
    if "NEGATIVE" in sentiment:
        return "Urgent"
    elif "POSITIVE" in sentiment:
        return "Friendly"
    else:
        return "Reminder"

# Function to extract sender’s name intelligently
def extract_sender(text):
    # Look for common sign-offs
    sender_match = re.search(r'\n(Best regards|Thanks|Sincerely|Regards|Cheers),?\s*\n*(.+)', text, re.IGNORECASE)
    if sender_match:
        return sender_match.group(2).strip()

    # Alternative: Extract last non-empty line as fallback
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.lower().startswith(("subject:", "hi", "hello", "dear", "team,", "i hope", "please", "let me know")):
            return line.strip()
    
    return "Unknown Sender"

# Function to read email file and generate an optimized subject
def generate_subject_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            email_text = file.read().strip()
        print(f"Email Text: {email_text}")  # Debugging line
        
        # Extract subject and body
        lines = email_text.split("\n")
        print(f"Lines: {lines}")  # Debugging line
        
        original_subject = lines[0].replace("Subject:", "").strip()
        email_body = " ".join(lines[1:]).strip()

        # Extract important dates
        extracted_dates = extract_dates(email_body)
        
        # Detect email tone
        email_tone = detect_tone(email_body)

        # Extract sender's name wisely
        sender = extract_sender(email_text)

        # Summarize email content using T5 model
        input_text = f"summarize: {original_subject} {email_body}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Trim to 12 words for clarity
        summary = " ".join(summary.split()[:12])

        # Remove unnecessary words (like 'let', 'is', 'it', etc.)
        filtered_summary = " ".join([word for word in summary.split() if word.lower() not in ["let", "is", "it"]])

        # Append key dates naturally
        if extracted_dates:
            if "deadline" in filtered_summary.lower():
                filtered_summary += f" {extracted_dates[-1]}"
            elif "review" in filtered_summary.lower():
                filtered_summary += f" by {extracted_dates[-1]}"

        return f"{sender} | {email_tone} | {filtered_summary}"

    except Exception as e:
        print(f"Error: {e}")  # Debugging line
        return f"Error processing file: {e}"

# Flask route for file upload
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Log file details
        print(f"Uploaded file: {file.filename}")
        
        # Save the uploaded file temporarily
        file_path = "temp_email.txt"
        file.save(file_path)

        # Generate optimized subject from the uploaded email file
        optimized_subject = generate_subject_from_file(file_path)
        return jsonify({"optimized_subject": optimized_subject})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
