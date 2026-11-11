import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("D:\TEC Project\ISEAR_dataset_7.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Use your columns
df = df[['content', 'sentiment']]

# Rename for consistency
df.columns = ['Sentence', 'Emotion']

print("✅ Dataset Loaded Successfully")
print(df.head())


# ==============================
# 2. TEXT CLEANING
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['Sentence'] = df['Sentence'].apply(clean_text)


# ==============================
# 3. ENCODE LABELS
# ==============================

le = LabelEncoder()
df['Emotion'] = le.fit_transform(df['Emotion'])

# Save label encoder
joblib.dump(le, "label_encoder.pkl")


# ==============================
# 4. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    df['Sentence'],
    df['Emotion'],
    test_size=0.2,
    random_state=42
)


# ==============================
# 5. TOKENIZATION
# ==============================

vocab_size = 10000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Save tokenizer
joblib.dump(tokenizer, "tokenizer.pkl")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')


# ==============================
# 6. BUILD CNN + BiGRU MODEL 🔥
# ==============================

input_layer = Input(shape=(max_len,))

embedding = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)

cnn = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
cnn = MaxPooling1D(pool_size=2)(cnn)

bigru = Bidirectional(GRU(64, return_sequences=False))(cnn)

dense = Dense(64, activation='relu')(bigru)
drop = Dropout(0.5)(dense)

output = Dense(7, activation='softmax')(drop)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("✅ Model Built Successfully")
model.summary()


# ==============================
# 7. TRAIN MODEL
# ==============================

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_pad, y_test)
)


# ==============================
# 8. SAVE MODEL
# ==============================

model.save("cnn_bigru_model.h5")

print("🎉 Model Training Completed & Saved as cnn_bigru_model.h5")