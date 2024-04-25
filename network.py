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
                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
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
  def __init__(self, num_actions: int, num_hidden_units: int, use_action_history: bool):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    self.use_action_history = use_action_history
    
    latent_dim = int(512)
    self.CVAE = CVAE(latent_dim)

    self.lstm_obs = layers.LSTM(256, return_sequences=True, return_state=True, kernel_regularizer='l2')

    self.conv_1_his = layers.Conv2D(8, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_2_his = layers.Conv2D(16, 2, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_3_his = layers.Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.lstm_his = layers.LSTM(256, return_sequences=True, return_state=True, kernel_regularizer='l2')
    
    self.common_1 = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.common_2 = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.common_3 = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')

    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')
    self.critic = layers.Dense(1, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_actions': self.num_actions,
        'num_hidden_units': self.num_hidden_units
    })

    return config
    
  def call(self, obs: tf.Tensor, action_history: tf.Tensor, memory_state_obs: tf.Tensor, carry_state_obs: tf.Tensor, 
           memory_state_his: tf.Tensor, carry_state_his: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(obs)[0]

    mean, logvar = self.CVAE.encode(obs)
    cvae_output = tf.concat((mean, logvar), axis=1)
    cvae_output_reshaped = layers.Reshape((16,64))(cvae_output)
    initial_state_obs = (memory_state_obs, carry_state_obs)
    lstm_output_obs, memory_state_obs, carry_state_obs = self.lstm_obs(cvae_output_reshaped, initial_state=initial_state_obs, 
                                                                       training=training)

    conv_1_his = self.conv_1_his(action_history)
    conv_2_his = self.conv_2_his(conv_1_his)
    conv_3_his = self.conv_3_his(conv_2_his)
    conv_3_his_reshaped = layers.Reshape((16,98))(conv_3_his)

    initial_state_his = (memory_state_his, carry_state_his)
    lstm_output_his, memory_state_his, carry_state_his = self.lstm_his(conv_3_his_reshaped, initial_state=initial_state_his, 
                                                                       training=training)

    X_input_obs = layers.Flatten()(lstm_output_obs)
    X_input_obs = self.common_1(X_input_obs)

    X_input_his = layers.Flatten()(lstm_output_his)
    X_input_his = self.common_2(X_input_his)

    if self.use_action_history:
      X_input = tf.concat([X_input_obs, X_input_his], 1)
    else:
      X_input = X_input_obs
    
    x = self.common_3(X_input)

    z = self.CVAE.reparameterize(mean, logvar)
    x_logit = self.CVAE.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=obs)
    
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    cvae_loss = logpx_z + logpz - logqz_x
    
    return self.actor(x), self.critic(x), memory_state_obs, carry_state_obs, memory_state_his, carry_state_his, cvae_loss



class InverseActionPolicy(tf.keras.Model):
  """Inverse Dynamics  network."""
  def __init__(self, num_actions: int, num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions

    # obs
    self.conv3d_1 = layers.Conv3D(filters=16, kernel_size=(5, 5, 5), padding="same")
    self.conv3d_2 = layers.Conv3D(filters=16, kernel_size=(3, 3, 3), padding="same")
    self.conv3d_3 = layers.Conv3D(filters=16, kernel_size=(1, 1, 1), padding="same")
    self.lstm_obs = layers.LSTM(512, return_sequences=True, return_state=True, kernel_regularizer='l2')
    self.common_obs = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')

    # his
    self.conv_1_his = layers.Conv2D(8, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_2_his = layers.Conv2D(16, 2, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_3_his = layers.Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.lstm_his = layers.LSTM(128, return_sequences=True, return_state=True, kernel_regularizer='l2')
    self.common_his = layers.Dense(num_hidden_units / 4, activation="relu", kernel_regularizer='l2')

    self.layer_normalization = layers.LayerNormalization()
    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')

    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({'num_actions': self.num_actions, 'num_hidden_units': self.num_hidden_units})

    return config
    
  def call(self, observation: tf.Tensor, action_history: tf.Tensor, memory_state_obs: tf.Tensor, carry_state_obs: tf.Tensor, 
           memory_state_his: tf.Tensor, carry_state_his: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # inputs.shape:  (1, 8, 64, 64, 3)
    print("observation.shape: ", observation.shape)
    print("action_history.shape: ", action_history.shape)

    batch_size = observation.shape[0]
    time_step = observation.shape[1]

    # obs
    conv3d_1 = self.conv3d_1(observation)
    conv3d_1 = layers.LayerNormalization()(conv3d_1)
    conv3d_1 = layers.ReLU()(conv3d_1)

    conv3d_2 = self.conv3d_2(conv3d_1)
    conv3d_2 = layers.LayerNormalization()(conv3d_2)
    conv3d_2 = layers.ReLU()(conv3d_2)

    conv3d_3 = self.conv3d_3(conv3d_2)
    conv3d_3 = layers.LayerNormalization()(conv3d_3)
    conv3d_3 = layers.ReLU()(conv3d_3)

    conv3d_reshaped = tf.reshape(conv3d_3, [batch_size, time_step, -1])

    initial_state_obs = (memory_state_obs, carry_state_obs)
    lstm_output_obs, final_memory_state, final_carry_state  = self.lstm_obs(conv3d_reshaped, initial_state=initial_state_obs, 
                                                                            training=training)

    X_input_obs = layers.Flatten()(conv3d_reshaped)
    X_input_obs = self.common_obs(X_input_obs)

    # act history
    conv_1_his = self.conv_1_his(action_history)
    conv_2_his = self.conv_2_his(conv_1_his)
    conv_3_his = self.conv_3_his(conv_2_his)
    conv_3_his_reshaped = layers.Reshape((16,98))(conv_3_his)

    initial_state_his = (memory_state_his, carry_state_his)
    lstm_output_his, memory_state_his, carry_state_his = self.lstm_his(conv_3_his_reshaped, initial_state=initial_state_his, 
                                                                       training=training)

    X_input_his = layers.Flatten()(lstm_output_his)
    X_input_his = self.common_his(X_input_his)

    # common
    print("X_input_obs.shape: ", X_input_obs.shape)
    print("X_input_his.shape: ", X_input_his.shape)
    print("")

    X_input = tf.concat([X_input_obs, X_input_his], 1)

    pi_latent  = self.actor(X_input)
    
    return pi_latent, final_memory_state, final_carry_state