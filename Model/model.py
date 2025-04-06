import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Concatenate, Dense, Input, Layer,
                                     MultiHeadAttention)
from tensorflow.keras.models import Model


class CrossAttention(Layer):
    def __init__(self, latent_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.attention_s2s = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim//num_heads
        )
        self.attention_shared2specific = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim//num_heads
        )
        self.gate_specific = tf.Variable(0.1, trainable=True, dtype=tf.float32)
        self.gate_shared = tf.Variable(0.1, trainable=True, dtype=tf.float32)

    def call(self, specific, shared):
        specific = tf.expand_dims(specific, 1)
        shared = tf.expand_dims(shared, 1)

        attn_s2s = self.attention_s2s(shared, specific, specific)
        shared_updated = shared + self.gate_shared * attn_s2s

        attn_shared2specific = self.attention_shared2specific(specific, shared, shared)
        specific_updated = specific + self.gate_specific * attn_shared2specific

        return tf.squeeze(specific_updated, 1), tf.squeeze(shared_updated, 1)

class FAKE_NEWS_DETECTOR:
    def __init__(self, input_d, domain_emb_d, latent_d, lambda1, lambda2, lambda3, lambda4=0.1, lambda5=0.1):
        self.input_shape = input_d
        self.latent_dim = latent_d
        self.no_domains = domain_emb_d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5

        self.cross_attention = CrossAttention(latent_d//2)
        self._build_models()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_specific_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.disc_shared_optimizer = tf.keras.optimizers.Adam(1e-5)

    def _build_models(self):
        input_layer = Input(shape=(self.input_shape,))

        specific = Dense(self.latent_dim//2, activation='relu')(input_layer)
        shared = Dense(self.latent_dim//2, activation='relu')(input_layer)

        specific_attn, shared_attn = self.cross_attention(specific, shared)
        latent_concat = Concatenate()([specific_attn, shared_attn])

        main_pred = Dense(1, activation='sigmoid')(latent_concat)
        decoded = Dense(self.input_shape)(Dense(self.latent_dim//2, activation='relu')(latent_concat))

        self.generator = Model(input_layer, [main_pred, decoded, specific_attn, shared_attn])

        self.domain_classifier_specific = self._build_discriminator()
        self.domain_classifier_shared = self._build_discriminator()

    def _build_discriminator(self):
        inp = Input(shape=(self.latent_dim//2,))
        x = Dense(self.no_domains * 2, activation='relu')(inp)
        out = Dense(self.no_domains, activation='sigmoid')(x)
        return Model(inp, out)

    def train_step(self, X_batch, y_batch, domain_batch):
        domain_indices = tf.argmax(domain_batch, axis=-1)

        # 1. Train Specific Domain Classifier
        with tf.GradientTape() as tape:
            _, _, specific, _ = self.generator(X_batch, training=False)
            pred = self.domain_classifier_specific(specific)
            loss = tf.reduce_mean(tf.square(domain_batch - pred))

        grads = tape.gradient(loss, self.domain_classifier_specific.trainable_variables)
        self.disc_specific_optimizer.apply_gradients(
            zip(grads, self.domain_classifier_specific.trainable_variables)
        )

        # 2. Train Shared Domain Classifier (Normal Training)
        with tf.GradientTape() as tape:
            _, _, _, shared = self.generator(X_batch, training=False)
            pred = self.domain_classifier_shared(shared)
            loss = tf.reduce_mean(tf.square(domain_batch - pred))

        grads = tape.gradient(loss, self.domain_classifier_shared.trainable_variables)
        self.disc_shared_optimizer.apply_gradients(
            zip(grads, self.domain_classifier_shared.trainable_variables)
        )

        # 3. Train Generator with Adversarial Component
        with tf.GradientTape() as tape: 
            preds, recon, specific, shared = self.generator(X_batch, training=True)
            preds = tf.squeeze(preds, axis=-1)

            
            recon_loss = tf.reduce_mean(tf.square(X_batch - recon))
            cls_loss = tf.keras.losses.binary_crossentropy(y_batch, preds)

            
            domain_specific_pred = self.domain_classifier_specific(specific)
            domain_shared_pred = self.domain_classifier_shared(shared)
            domain_loss_specific = tf.reduce_mean(tf.square(domain_batch - domain_specific_pred))
            domain_loss_shared = tf.reduce_mean(tf.square(domain_batch - domain_shared_pred))

            ortho_loss = tf.reduce_mean(tf.square(tf.reduce_sum(specific * shared, axis=-1)))
            domain_matrix = tf.cast(tf.equal(domain_indices[:, None], domain_indices[None, :]), tf.float32)
            contrast_loss = tf.reduce_mean(domain_matrix * tf.matmul(specific, shared, transpose_b=True))

            total_loss = (cls_loss +
                         self.lambda1 * recon_loss +
                         self.lambda2 * domain_loss_specific -
                         self.lambda3 * domain_loss_shared +
                         self.lambda4 * ortho_loss +
                         self.lambda5 * contrast_loss)

        grads = tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {
            'cls_loss': cls_loss.numpy().mean(),
            'recon_loss': recon_loss.numpy(),
            'domain_loss_specific': domain_loss_specific.numpy(),
            'domain_loss_shared': domain_loss_shared.numpy(),
            'ortho_loss': ortho_loss.numpy(),
            'contrast_loss': contrast_loss.numpy()
        }

    def train(self, X_train, y_train, yd_train, epochs, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train, yd_train)
        ).shuffle(1024).batch(batch_size)

        for epoch in range(epochs):
            epoch_losses = []
            for batch in dataset:
                losses = self.train_step(*batch)
                epoch_losses.append(losses)

            avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
            print(f"\nEpoch {epoch+1}:")
            print(f"Classification Loss: {avg_losses['cls_loss']:.4f}")
            print(f"Reconstruction Loss: {avg_losses['recon_loss']:.4f}")
            print(f"Specific Domain Loss: {avg_losses['domain_loss_specific']:.4f}")
            print(f"Shared Domain Loss: {avg_losses['domain_loss_shared']:.4f}")
            print(f"Orthogonality Loss: {avg_losses['ortho_loss']:.4f}")
            print(f"Contrastive Loss: {avg_losses['contrast_loss']:.4f}")

    def evaluate(self, X_test, y_test, yd_test):
        preds, _, _, _ = self.generator(X_test)
        preds = tf.squeeze(preds, axis=-1)
        y_pred = tf.cast(preds > 0.5, tf.float32)

        accuracy = tf.keras.metrics.BinaryAccuracy()(y_test, y_pred)
        precision = tf.keras.metrics.Precision()(y_test, y_pred)
        recall = tf.keras.metrics.Recall()(y_test, y_pred)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return {
        'accuracy': accuracy.numpy(),
        'precision': precision.numpy(),
        'recall': recall.numpy(),
        'f1': f1.numpy()
        }