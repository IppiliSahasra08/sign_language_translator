# Compare with your LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Your LSTM with raw data
X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
)

model_lstm = Sequential([
    LSTM(32, return_sequences=True, activation='relu', input_shape=(10, 126)),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(32, return_sequences=False, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model_lstm.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_lstm.fit(X_train_lstm, Y_train_lstm, epochs=50, validation_data=(X_test_lstm, Y_test_lstm), verbose=0)

lstm_accuracy = model_lstm.evaluate(X_test_lstm, Y_test_lstm, verbose=0)[1]
print(f"\nLSTM (Sequential) Accuracy: {lstm_accuracy:.4f}")