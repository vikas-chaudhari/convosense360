import re
import emoji
from transformers import pipeline
from langdetect import detect

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")


# Function to clean WhatsApp chat data and extract messages
def extract_messages_from_file(file_path):
    messages = []

    # WhatsApp message format regex (can vary depending on the language/format)
    pattern = r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} (AM|PM) - [^:]+: (.*)'

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            match = re.match(pattern, line)
            if match:
                message = match.groups()[1]
                # Remove emoji if needed (useful for Hinglish)
                message_clean = emoji.demojize(message)
                messages.append(message_clean)

    return messages


# Function to detect language (Hinglish, Hindi, English)
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


# Function to analyze sentiment for each message
def analyze_chat_sentiment(messages):
    sentiments = []

    for message in messages:
        # Detect language and apply sentiment analysis
        lang = detect_language(message)

        if lang in ['hi', 'en']:  # Proceed if detected language is Hindi or English
            sentiment = sentiment_pipeline(message)[0]
            sentiments.append({
                'message': message,
                'sentiment': sentiment['label'],
                'confidence': sentiment['score']
            })

    return sentiments


# Main function
def process_whatsapp_chat(file_path):
    # Step 1: Extract messages from the WhatsApp chat file
    messages = extract_messages_from_file(file_path)

    # Step 2: Analyze sentiment of each message
    sentiments = analyze_chat_sentiment(messages)

    # Step 3: Output the sentiment results
    for sentiment_data in sentiments:
        print(
            f"Message: {sentiment_data['message']}\nSentiment: {sentiment_data['sentiment']}, Confidence: {sentiment_data['confidence']:.4f}\n")


# Example usage
file_path = '../WhatsApp Chat with TechSaksham SKNCOE\WhatsApp Chat with TechSaksham SKNCOE.txt'  # Replace this with the path to your WhatsApp chat file
process_whatsapp_chat(file_path)

