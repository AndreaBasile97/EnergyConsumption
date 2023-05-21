import tensorflow as tf

# Creating the model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, activation='relu', input_shape=(1, 204), return_sequences = True),
        tf.keras.layers.Dropout(0.2),  # 20% dropout
        tf.keras.layers.LSTM(64, activation='relu', return_sequences = True),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences = True),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences = True),
        tf.keras.layers.Dropout(0.2),  # 20% dropout
        tf.keras.layers.LSTM(128, activation='relu'),
        tf.keras.layers.Dense(16)  # add this line for L2 regularization
    ])
    return model

# Loss function
def RSE(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true)
    squared_error_num = tf.reduce_sum(tf.square(y_true - y_pred))
    squared_error_den = tf.reduce_sum(tf.square(y_true - true_mean))
    rse_loss = squared_error_num / squared_error_den
    return rse_loss

# R^2 metric
def r2(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# Compilation of the model
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=[RSE, r2, tf.keras.metrics.RootMeanSquaredError()])
    return model

# Train and make prediction
def train_and_predict(model, X_train_reshaped, y_train, X_test_reshaped):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',  # monitor training loss
        patience=3,  # stop training if loss does not improve for 3 consecutive epochs
        restore_best_weights=True  # restore the best weights from the epoch with the best loss
    )
    model.fit(
        X_train_reshaped,
        y_train.to_numpy(),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],  # pass the callback to the fit function
        verbose=2
    )
    y_pred = model.predict(X_test_reshaped)
    return y_pred


# Compute metrics
def compute_metrics(y_test, y_pred):
    rmse = tf.keras.metrics.RootMeanSquaredError()
    rmse.update_state(y_test.to_numpy(), y_pred)
    rse = RSE(y_test.to_numpy(), y_pred)
    r2_score = r2(y_test.to_numpy(), y_pred)
    return rmse.result().numpy(), rse.numpy(), r2_score.numpy()


