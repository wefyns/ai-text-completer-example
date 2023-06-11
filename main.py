import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
data_path = "text_corpus.txt"
with open(data_path, "r") as file:
    data = file.read()

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences using list of tokens
input_sequences = []
for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Prepare training data
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X_train = input_sequences[:, :-1]
y_train = input_sequences[:, -1]

# Transform y_train to one-hot encoding format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=total_words)

# Сreate model RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
    tf.keras.layers.LSTM(units=150),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Check if model weights exist
weights_path = "model_weights.h5"
if os.path.exists(weights_path):
    # Load model weights
    model.load_weights(weights_path)
else:
    # Train model
    model.fit(X_train, y_train, epochs=100, verbose=1)
    # Save model weights
    model.save_weights(weights_path)

# Генерация текста
def generate_text(seed_text, num_words):
    for _ in range(num_words):
        encoded_text = tokenizer.texts_to_sequences([seed_text])[0]
        encoded_text = pad_sequences([encoded_text], maxlen=max_sequence_len-1, padding='pre')
        predicted_word_index = np.argmax(model.predict(encoded_text), axis=-1)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        seed_text += " " + predicted_word
    return seed_text

# Generate text
seed_text = "The quick brown"
generated_text = generate_text(seed_text, 5)
print(generated_text)
