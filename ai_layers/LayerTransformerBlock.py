import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerEncoderBlock(layers.Layer):
    """Implements a single encoder block from the Transformer model architecture."""
	
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        """
		Implements a Transformer encoder as per the Attention Is All You Need paper.
		embed_dim: The number of units to use in the embedding dimension.
		dense_dim: The number of units in the internal dense layer.
		num_heads: The number of attention heads to create.
		"""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        """Runs the given inputs through the model."""
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1
