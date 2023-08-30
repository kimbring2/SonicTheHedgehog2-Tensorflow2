import time
import math
import zmq
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTMCell
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import threading
import random
import collections
import argparse
from absl import flags
from absl import logging
from typing import Any, List, Sequence, Tuple
from gym.spaces import Dict, Discrete, Box, Tuple
import network
import cv2
from parametric_distribution import get_parametric_distribution_for_action_space

parser = argparse.ArgumentParser(description='Sonic IMPALA Server')
parser.add_argument('--env_num', type=int, default=2, help='ID of environment')
parser.add_argument('--gpu_use', action='store_false', help='use gpu')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
arguments = parser.parse_args()

tfd = tfp.distributions

if gpu_use:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


socket_list = []
for i in range(0, arguments.env_num):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(6555 + i))

    socket_list.append(socket)

    
num_actions = 23
state_size = (84,84,3)  
act_history_size = (23,23,1)  

batch_size = 1

unroll_length = 101
queue = tf.queue.FIFOQueue(1, dtypes=[tf.int32, tf.float32, tf.bool, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32], 
                           shapes=[[unroll_length+1],[unroll_length+1],[unroll_length+1],[unroll_length+1,*state_size],
                                   [unroll_length+1,num_actions],[unroll_length+1],[unroll_length+1,*act_history_size],
                                   [unroll_length+1,256],[unroll_length+1,256],[unroll_length+1,256],[unroll_length+1,256]
                                  ])
Unroll = collections.namedtuple('Unroll', 'env_id reward done observation policy action action_history memory_state_obs carry_state_obs memory_state_his carry_state_his')

num_hidden_units = 1024
model = network.ActorCritic(num_actions, num_hidden_units)
sl_model = network.ActorCritic(num_actions, num_hidden_units)

if arguments.pretrained_model != None:
    print("Load Pretrained Model")
    sl_model.load_weights("model/" + arguments.pretrained_model)
    model.load_weights("model/" + arguments.pretrained_model)
    #model.load_weights("model/" + "reinforcement_model_27500")
    
#model.set_weights(sl_model.get_weights())

num_action_repeats = 1
total_environment_frames = int(4e7)

iter_frame_ratio = (batch_size * unroll_length * num_action_repeats)
final_iteration = int(math.ceil(total_environment_frames / iter_frame_ratio))
    
lr = tf.keras.optimizers.schedules.PolynomialDecay(0.0001, final_iteration, 0)
optimizer = tf.keras.optimizers.Adam(lr)


def take_vector_elements(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(num_actions))
kl = tf.keras.losses.KLDivergence()

def update(states, actions, agent_policies, rewards, dones, act_histories,
           memory_states_obs, carry_states_obs, memory_states_his, carry_states_his):
    states = tf.transpose(states, perm=[1, 0, 2, 3, 4])
    actions = tf.transpose(actions, perm=[1, 0])
    agent_policies = tf.transpose(agent_policies, perm=[1, 0, 2])
    rewards = tf.transpose(rewards, perm=[1, 0])
    dones = tf.transpose(dones, perm=[1, 0])
    act_histories = tf.transpose(act_histories, perm=[1, 0, 2, 3, 4])
    memory_states_obs = tf.transpose(memory_states_obs, perm=[1, 0, 2])
    carry_states_obs = tf.transpose(carry_states_obs, perm=[1, 0, 2])
    memory_states_his = tf.transpose(memory_states_his, perm=[1, 0, 2])
    carry_states_his = tf.transpose(carry_states_his, perm=[1, 0, 2])
    
    batch_size = states.shape[0]
        
    online_variables = model.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(online_variables)
               
        learner_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sl_learner_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sl_learner_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        cvae_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sl_cvae_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        memory_state_obs = memory_states_obs[0]
        carry_state_obs = carry_states_obs[0]
        sl_memory_state_obs = memory_states_obs[0]
        sl_carry_state_obs = carry_states_obs[0]

        memory_state_his = memory_states_his[0]
        carry_state_his = carry_states_his[0]
        sl_memory_state_his = memory_states_his[0]
        sl_carry_state_his = carry_states_his[0]
        for i in tf.range(0, batch_size):
            prediction = model(states[i], act_histories[i], memory_state_obs, carry_state_obs,
                               memory_state_his, carry_state_his, training=True)
            sl_prediction = sl_model(states[i], act_histories[i], sl_memory_state_obs, sl_carry_state_obs, 
                                     sl_memory_state_his, sl_carry_state_his, training=True)

            learner_policies = learner_policies.write(i, prediction[0])
            learner_values = learner_values.write(i, prediction[1])
            
            sl_learner_policies = sl_learner_policies.write(i, sl_prediction[0])
            sl_learner_values = sl_learner_values.write(i, sl_prediction[1])
            
            memory_state_obs = prediction[2]
            carry_state_obs = prediction[3]
            memory_state_his = prediction[4]
            carry_state_his = prediction[5]
            cvae_loss = prediction[6]

            sl_memory_state_obs = sl_prediction[2]
            sl_carry_state_obs = sl_prediction[3]
            sl_memory_state_his = sl_prediction[4]
            sl_carry_state_his = sl_prediction[5]
            sl_cvae_loss = sl_prediction[6]

            cvae_losses = cvae_losses.write(i, cvae_loss)
            sl_cvae_losses = sl_cvae_losses.write(i, sl_cvae_loss)

        learner_policies = learner_policies.stack()
        learner_values = learner_values.stack()
        sl_learner_policies = sl_learner_policies.stack()
        sl_learner_values = sl_learner_values.stack()

        cvae_losses = cvae_losses.stack()
        sl_cvae_losses = sl_cvae_losses.stack()

        learner_policies = tf.reshape(learner_policies, [states.shape[0], states.shape[1], -1])
        learner_values = tf.reshape(learner_values, [states.shape[0], states.shape[1], -1])
        sl_learner_policies = tf.reshape(sl_learner_policies, [states.shape[0], states.shape[1], -1])
        sl_learner_values = tf.reshape(sl_learner_values, [states.shape[0], states.shape[1], -1])
        
        dist = tfd.Categorical(logits=learner_policies[:-1])
        sl_dist = tfd.Categorical(logits=sl_learner_policies[:-1])
        kl_loss = tfd.kl_divergence(dist, sl_dist)
        kl_loss = 0.0001 * tf.reduce_mean(kl_loss)
        
        agent_logits = tf.nn.softmax(agent_policies[:-1])
        actions = actions[:-1]
        rewards = rewards[1:]
        dones = dones[1:]
        
        learner_logits = tf.nn.softmax(learner_policies[:-1])
        
        learner_values = tf.squeeze(learner_values, axis=2)
        
        bootstrap_value = learner_values[-1]
        learner_values = learner_values[:-1]
        
        discounting = 0.99
        discounts = tf.cast(~dones, tf.float32) * discounting

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        
        target_action_log_probs = parametric_action_distribution.log_prob(learner_policies[:-1], actions)
        behaviour_action_log_probs = parametric_action_distribution.log_prob(agent_policies[:-1], actions)
        
        lambda_ = 1.0
        
        log_rhos = target_action_log_probs - behaviour_action_log_probs
        
        log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
        discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        values = tf.convert_to_tensor(learner_values, dtype=tf.float32)
        bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
        
        clip_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
        clip_pg_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
        
        rhos = tf.math.exp(log_rhos)
        
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
        
        cs = tf.minimum(1.0, rhos, name='cs')
        cs *= tf.convert_to_tensor(lambda_, dtype=tf.float32)

        values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
        
        acc = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = []
        for i in range(int(discounts.shape[0]) - 1, -1, -1):
            discount, c, delta = discounts[i], cs[i], deltas[i]
            acc = delta + discount * c * acc
            vs_minus_v_xs.append(acc)  
            
        vs_minus_v_xs = vs_minus_v_xs[::-1]
            
        vs = tf.add(vs_minus_v_xs, values, name='vs')
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
            
        pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
            
        vs = tf.stop_gradient(vs)
        pg_advantages = tf.stop_gradient(pg_advantages)
            
        actor_loss = target_action_log_probs * pg_advantages
        actor_loss = -tf.reduce_mean(actor_loss)
            
        baseline_cost = 0.1
        v_error = values - vs
        critic_loss = tf.square(v_error)
        critic_loss = baseline_cost * 0.5 * tf.reduce_mean(critic_loss)
            
        entropy_loss = parametric_action_distribution.entropy(learner_policies[:-1])
        entropy_loss = tf.reduce_mean(entropy_loss)
        entropy_loss = 0.0025 * -entropy_loss

        cvae_loss = -tf.reduce_mean(cvae_losses)
        
        total_loss = actor_loss + critic_loss + entropy_loss + kl_loss + cvae_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss


@tf.function
def prediction(state, act_history, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his):
    prediction = model(state, act_history, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his, 
                       training=False)
    dist = tfd.Categorical(logits=prediction[0])
    action = int(dist.sample()[0])
    policy = prediction[0]

    memory_state_obs = prediction[2]
    carry_state_obs = prediction[3]
    memory_state_his = prediction[4]
    carry_state_his = prediction[5]

    mean, logvar = model.CVAE.encode(state)
    z = model.CVAE.reparameterize(mean, logvar)
    predictions = model.CVAE.sample(z)

    return action, policy, memory_state_obs, carry_state_obs, memory_state_his, carry_state_his, predictions


@tf.function
def enque_data(env_ids, rewards, dones, states, policies, actions, act_histories, memory_states_obs, carry_states_obs,
               memory_states_his, carry_states_his):
    queue.enqueue((env_ids, rewards, dones, states, policies, actions, act_histories, memory_states_obs, carry_states_obs, 
                   memory_states_his, carry_states_his))


def Data_Thread(coord, i):
    env_ids = np.zeros((unroll_length + 1), dtype=np.int32)
    states = np.zeros((unroll_length + 1, *state_size), dtype=np.float32)
    actions = np.zeros((unroll_length + 1), dtype=np.int32)
    policies = np.zeros((unroll_length + 1, num_actions), dtype=np.float32)
    rewards = np.zeros((unroll_length + 1), dtype=np.float32)
    dones = np.zeros((unroll_length + 1), dtype=np.bool)
    act_histories = np.zeros((unroll_length + 1, *act_history_size), dtype=np.float32)
    memory_states_obs = np.zeros((unroll_length + 1, 256), dtype=np.float32)
    carry_states_obs = np.zeros((unroll_length + 1, 256), dtype=np.float32)
    memory_states_his = np.zeros((unroll_length + 1, 256), dtype=np.float32)
    carry_states_his = np.zeros((unroll_length + 1, 256), dtype=np.float32)

    memory_index = 0

    index = 0

    min_elapsed_time = 5.0

    reward_list = []

    while not coord.should_stop(): 
        start = time.time()

        message = socket_list[i].recv_pyobj()
        if memory_index == unroll_length:
            enque_data(env_ids, rewards, dones, states, policies, actions, act_histories,
                       memory_states_obs, carry_states_obs, memory_states_his, carry_states_his)

            env_ids[0] = env_ids[memory_index]
            states[0] = states[memory_index]
            actions[0] = actions[memory_index]
            policies[0] = policies[memory_index]
            rewards[0] = rewards[memory_index]
            dones[0] = dones[memory_index]
            act_histories[0] = act_histories[memory_index]
            memory_states_obs[0] = memory_states_obs[memory_index]
            carry_states_obs[0] = carry_states_obs[memory_index]
            memory_states_his[0] = memory_states_his[memory_index]
            carry_states_his[0] = carry_states_his[memory_index]

            memory_index = 1

        state = tf.constant(np.array(message["observation"]))
        act_history = tf.constant(np.array(message["act_history"]))
        memory_state_obs = tf.constant(message["memory_state_obs"])
        carry_state_obs = tf.constant(message["carry_state_obs"])
        memory_state_his = tf.constant(message["memory_state_his"])
        carry_state_his = tf.constant(message["carry_state_his"])

        action, policy, new_memory_state_obs, new_carry_state_obs, new_memory_state_his, new_carry_state_his, predictions = prediction(state, 
                                                                                                                                       act_history,
                                                                                                                                       memory_state_obs, 
                                                                                                                                       carry_state_obs,
                                                                                                                                       memory_state_his, 
                                                                                                                                       carry_state_his)

        #print("state.numpy(): ", state.numpy())
        #print("predictions.shape: ", predictions.shape)
        #cv2.imshow("state", state.numpy())
        #cv2.imshow("predictions", predictions)
        #cv2.waitKey(1)

        env_ids[memory_index] = message["env_id"]
        states[memory_index] = message["observation"]
        actions[memory_index] = action
        policies[memory_index] = policy
        rewards[memory_index] = message["reward"]
        dones[memory_index] = message["done"]
        act_histories[memory_index] = message["act_history"]
        memory_states_obs[memory_index] = memory_state_obs
        carry_states_obs[memory_index] = carry_state_obs
        memory_states_his[memory_index] = memory_state_his
        carry_states_his[memory_index] = carry_state_his

        reward_list.append(message["reward"])

        socket_list[i].send_pyobj({"env_id": message["env_id"], "action": action, 
                                   "memory_state_obs": new_memory_state_obs, "carry_state_obs": new_carry_state_obs,
                                   "memory_state_his": new_memory_state_his, "carry_state_his": new_carry_state_his})

        memory_index += 1
        index += 1
        if index % 200 == 0:
            average_reward = sum(reward_list[-50:]) / len(reward_list[-50:])
            #print("average_reward: ", average_reward)

        end = time.time()
        elapsed_time = end - start

    if index == 100000000:
        coord.request_stop()


unroll_queues = []
unroll_queues.append(queue)

def dequeue(ctx):
    dequeue_outputs = tf.nest.map_structure(
        lambda *args: tf.stack(args), 
        *[unroll_queues[ctx].dequeue() for i in range(batch_size)]
      )

    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and repack.
    return tf.nest.flatten(dequeue_outputs)


def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(_dequeue, num_parallel_calls=1)


if arguments.gpu_use == True:
    device_name = '/device:GPU:0'
else:
    device_name = '/device:CPU:0'

dataset = dataset_fn(0)
it = iter(dataset)


@tf.function
def minimize(iterator):
    dequeue_data = next(iterator)

    update(dequeue_data[3], dequeue_data[5], dequeue_data[4], dequeue_data[1], dequeue_data[2], dequeue_data[6], dequeue_data[7],
           dequeue_data[8], dequeue_data[9], dequeue_data[10])


def Train_Thread(coord):
    index = 0
    while not coord.should_stop():
        #print("index : ", index)
        index += 1

        minimize(it)
        #time.sleep(1)

        #if index % 2500 == 0:
        #    model.save_weights('model/reinforcement_model_' + str(index))

        if index == 100000000:
            coord.request_stop()


coord = tf.train.Coordinator(clean_stop_exception_types=None)

thread_data_list = []
for i in range(arguments.env_num):
    thread_data = threading.Thread(target=Data_Thread, args=(coord,i))
    thread_data_list.append(thread_data)

thread_train = threading.Thread(target=Train_Thread, args=(coord,))
thread_train.start()

for thread_data in thread_data_list:
    thread_data.start()

for thread_data in thread_data_list:
    coord.join(thread_data)

coord.join(thread_train)
