import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, initializers

# [CRITICAL] Import PADDING_VALUE to ensure Masking matches Data Engineering
from src.config import PADDING_VALUE


class AttentionBlock(layers.Layer):
    """
    Self-Attention mechanism for Time Series.
    Computes a weighted sum of the input sequence based on learned importance scores.

    Output: Context vector (2D tensor: [batch_size, units])
    """

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: Weight matrix for the attention score computation
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        # b: Bias term
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionBlock, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)

        # 1. Compute scores: e = tanh(dot(x, W) + b)
        # Result shape: (batch_size, time_steps, 1)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)

        # 2. Compute weights: alpha = softmax(e)
        # Weights sum to 1 across the time dimension
        a = tf.nn.softmax(e, axis=1)

        # 3. Weighted sum: context = sum(a * x)
        # Result shape: (batch_size, features)
        output = tf.reduce_sum(x * a, axis=1)

        return output


class FlightHyperModel(kt.HyperModel):
    """
    Research-grade Architectural Search Space for Flight Data Regression.

    Key Features:
    - Dynamic Masking (aligned with PADDING_VALUE)
    - Hybrid CNN-RNN Topology
    - Batch Normalization for convergence stability
    - Optional Self-Attention Mechanism
    - Orthogonal Initialization for Recurrent Layers
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = keras.Input(shape=self.input_shape)

        # 1. Masking (CRITICAL FIX)
        # Must match the padding value used in engineering.py (-1.0e9)
        # Using 0.0 here would cause the model to treat padding as massive valid signal.
        x = layers.Masking(mask_value=PADDING_VALUE)(inputs)

        # 2. CNN Block (Local Feature Extraction)
        if hp.Boolean("use_cnn_block"):
            x = layers.Conv1D(
                filters=hp.Int("cnn_filters", min_value=16, max_value=64, step=16),
                kernel_size=hp.Int("cnn_kernel_size", min_value=3, max_value=7, step=2),
                padding="same"
            )(x)
            # Batch Norm is best applied between Conv and Activation (or after)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            # Optional Pooling to reduce dimensionality
            if hp.Boolean("use_pooling"):
                x = layers.MaxPooling1D(pool_size=2)(x)

        # 3. Recurrent Block (Temporal Modeling)
        rnn_type = hp.Choice("rnn_type", ["LSTM", "GRU"])
        num_rnn_layers = hp.Int("num_rnn_layers", min_value=1, max_value=3)
        use_attention = hp.Boolean("use_attention")

        for i in range(num_rnn_layers):
            # Logic for return_sequences:
            # - Intermediate layers: ALWAYS True
            # - Last layer:
            #     - If Attention is ON: True (Attention needs the full sequence)
            #     - If Attention is OFF: False (Standard Many-to-One)
            is_last_layer = (i == num_rnn_layers - 1)

            if is_last_layer and not use_attention:
                return_sequences = False
            else:
                return_sequences = True

            units = hp.Int(f"rnn_units_{i}", min_value=32, max_value=256, step=32)

            # Configuration dict for the layer
            rnn_config = {
                "units": units,
                "return_sequences": return_sequences,
                "recurrent_initializer": "orthogonal"  # Best practice for RNNs
            }

            if rnn_type == "LSTM":
                x = layers.LSTM(**rnn_config)(x)
            else:
                x = layers.GRU(**rnn_config)(x)

            # Dropout for regularization
            dropout_rate = hp.Float(f"rnn_dropout_{i}", 0.0, 0.5, step=0.1)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

        # 4. Attention Mechanism (Optional)
        if use_attention:
            # Collapses the time dimension by weighted averaging
            x = AttentionBlock()(x)

        # 5. Dense Head
        if hp.Boolean("use_dense_head"):
            x = layers.Dense(
                units=hp.Int("dense_units", min_value=16, max_value=128, step=16)
            )(x)
            x = layers.BatchNormalization()(x)  # Stabilize dense inputs
            x = layers.Activation("relu")(x)

            head_dropout = hp.Float("head_dropout", 0.0, 0.5, step=0.1)
            if head_dropout > 0:
                x = layers.Dropout(head_dropout)(x)

        # 6. Output
        outputs = layers.Dense(1, activation="linear")(x)

        # 7. Compilation
        model = keras.Model(inputs=inputs, outputs=outputs, name="Flight_HyperModel_v2")

        lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae", "root_mean_squared_error"]
        )
        return model