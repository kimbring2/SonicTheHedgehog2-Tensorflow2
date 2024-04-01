import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(80, 80, 3)),
                layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Flatten(),
                layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=20*20*32, activation=tf.nn.relu),
                layers.Reshape(target_shape=(20, 20, 32)),
                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                                activation='relu'),
                layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                                activation='relu'),
                layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
    
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
    
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            
            return probs
    
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""
  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    
    self.conv_1 = layers.Conv2D(32, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_2 = layers.Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_3 = layers.Conv2D(64, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_4 = layers.Conv2D(512, 7, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    
    #self.latent_dim = 60
    #self.CVAE = CVAE(self.latent_dim)
    #self.CVAE_encoder = layers.Dense(256, activation="relu", kernel_regularizer='l2')

    self.lstm = layers.LSTM(256, return_sequences=True, return_state=True, kernel_regularizer='l2')
    
    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')
    self.critic = layers.Dense(1, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_actions': self.num_actions,
        'num_hidden_units': self.num_hidden_units
    })

    return config
    
  def call(self, obs: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor, 
           training) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(obs)[0]

    print("obs.shape: ", obs.shape)
    #pre_action = tf.expand_dims(pre_action, 0)
    #tf.print("pre_action.shape: ", pre_action.shape)

    #pre_action_tile = tf.ones(obs.shape, tf.float32) * (pre_action[0])
    #tf.print("pre_action_tile.shape: ", pre_action_tile.shape)

    #input_array = tf.concat([obs, pre_action_tile], 3)

    conv_1 = self.conv_1(obs)
    conv_2 = self.conv_2(conv_1)
    conv_3 = self.conv_3(conv_2)
    conv_4 = self.conv_4(conv_3)

    #mean, logvar = self.CVAE.encode(obs)
    #cvae_output = tf.concat((mean, logvar), axis=1)

    #tf.print("cvae_output.shape 1: ", cvae_output.shape)
    #tf.print("pre_action.shape: ", pre_action.shape)

    #cvae_output = tf.concat((cvae_output, pre_action), axis=1)
    #cvae_output_encoded = self.CVAE_encoder(cvae_output)

    print("conv_4.shape: ", conv_4.shape)
    
    conv_4_reshaped = layers.Reshape((16, 32))(conv_4)

    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state  = self.lstm(conv_4_reshaped, initial_state=initial_state, 
                                                                    training=training)
    
    X_input = layers.Flatten()(lstm_output)
    #tf.print("X_input.shape: ", X_input.shape)

    x = self.common(X_input)

    return self.actor(x), self.critic(x), final_memory_state, final_carry_state



class InverseActionPolicy(tf.keras.Model):
  """Inverse Dynamics  network."""
  def __init__(self, num_actions: int, num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    #self.input_shape =(1, 32, 84, 84, 3)

    self.conv2d_lstm_1 = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")
    self.batch_normalization_1 = layers.BatchNormalization()
    self.conv2d_lstm_2 = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")
    self.batch_normalization_2 = layers.BatchNormalization()
    self.conv2d_lstm_3 = layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding="same", return_sequences=True, activation="relu")
    self.conv3d = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")

    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({'num_actions': self.num_actions, 'num_hidden_units': self.num_hidden_units})

    return config
    
  def call(self, inputs: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor]:
    print("inputs.shape: ", inputs.shape)

    batch_size = tf.shape(inputs)[0]

    x = self.conv2d_lstm_1(inputs)
    print("x.shape 1: ", x.shape)

    x = self.batch_normalization_1(x)
    x = self.conv2d_lstm_2(x)
    x = self.batch_normalization_2(x)
    x = self.conv2d_lstm_3(x)
    x = self.conv3d(x)

    print("x.shape: ", x.shape)

    X_input = layers.Flatten()(x)
    x = self.common(X_input)
    
    return self.actor(x), self.actor(x)