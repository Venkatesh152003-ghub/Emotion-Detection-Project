from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ==============================
# 1. LOAD MODEL + FILES
# ==============================

model = load_model("cnn_bigru_model.h5")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

max_len = 100


# ==============================
# 2. TEXT PREPROCESSING
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# ==============================
# 3. RULE-BASED BOOST (OPTIONAL)
# ==============================

def rule_based(text):
    text_lower = text.lower()

    # Fear
    if any(word in text_lower for word in [
        'scared', 'afraid', 'fear', 'terrified', 'stress', 'stressed',
        'nervous', 'worried', 'anxious', 'panic', 'panicking',
        'tense', 'uneasy', 'overwhelmed',
        "can't handle", 'too much pressure'
    ]):
        return 'fear'

    # Joy
    elif any(word in text_lower for word in [
        'happy', 'joy', 'excited', 'glad', 'delighted',
        'pleased', 'satisfied', 'awesome', 'fantastic',
        'great', 'amazing', 'love'
    ]):
        return 'joy'

    # Sadness
    elif any(word in text_lower for word in [
        'sad', 'unhappy', 'depressed', 'down', 'cry',
        'upset', 'hurt', 'lonely', 'miserable',
        'heartbroken', 'tired',
        'giving up', 'feel alone', 'no hope'
    ]):
        return 'sadness'

    # Anger
    elif any(word in text_lower for word in [
        'angry', 'mad', 'furious', 'frustrated',
        'annoyed', 'irritated', 'rage',
        'pissed', 'hate', 'fed up', 'had enough'
    ]):
        return 'anger'

    # Disgust
    elif any(word in text_lower for word in [
        'disgust', 'disgusting', 'gross',
        'nasty', 'terrible', 'bad',
        'worst', 'awful'
    ]):
        return 'disgust'

    # Shame
    elif any(word in text_lower for word in [
        'ashamed', 'embarrassed', 'humiliated'
    ]):
        return 'shame'

    # Guilt
    elif any(word in text_lower for word in [
        'guilty', 'regret', 'sorry'
    ]):
        return 'guilt'

    else:
        return 'Neutral'

    


# ==============================
# 4. ROUTES
# ==============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # ✅ Step 1: Rule-based (fast)
        rule_result = rule_based(text)
        if rule_result:
            return jsonify({'emotion': rule_result})

        # ✅ Step 2: Deep Learning Prediction
        clean = clean_text(text)

        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')

        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction, axis=1)[0]

        emotion = label_encoder.inverse_transform([predicted_class])[0]

        return jsonify({
            'emotion': emotion,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================
# 5. RUN SERVER
# ==============================

if __name__ == '__main__':
    app.run(debug=True)