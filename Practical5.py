import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Sample text data
texts = [
    "I love it!",
    "It’s fine.",
    "Terrible experience.",
    "Really awesome service.",
    "Not good at all."
]

labels = [2, 1, 0, 2, 0]  # 0=Negative, 1=Neutral, 2=Positive

# Preprocess
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=5)
y = tf.keras.utils.to_categorical(labels, 3)

# LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=5),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, verbose=0)

# Predict sentiment
preds = model.predict(X)
predicted = np.argmax(preds, axis=1)



G = nx.Graph()

# Add nodes with predicted sentiment
for i, s in enumerate(predicted):
    G.add_node(i, sentiment=s)

# Add some edges (connections)
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

# Colors: 0=red, 1=gray, 2=green
colors = ['red' if G.nodes[n]['sentiment'] == 0 else
          'gray' if G.nodes[n]['sentiment'] == 1 else
          'green' for n in G.nodes]

nx.draw(G, with_labels=True, node_color=colors, node_size=800)
plt.title("Sentiment Graph (RNN-based)")
plt.show()
