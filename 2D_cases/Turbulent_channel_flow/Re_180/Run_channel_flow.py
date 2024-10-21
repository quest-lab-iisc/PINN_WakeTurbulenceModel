import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np

from keras.initializers import GlorotUniform

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.chdir('./')

from Channel_flow import PdeModel, get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
num_samples = 2000  #number of sample points
nu = 1/180.         #kinematic viscosity

#boundary and grid data
#xb, yb - boundary x and y values
#xd, yd - domain x and y values
#u,v - velocity boundary values
#p - pressure boundary values 
#k - TKE boundary values
#eps - epsilon boundary values
xb_top, yb_top, xb_bottom, yb_bottom, xb_left, yb_left, xb_right, yb_right, xd, yd, u, v, p, k, eps= get_ibc_and_inner_data(num_samples=num_samples, nu=nu)

#inputs and outputs
ivals = {'xin': xd, 'yin': yd, 'xb_top': xb_top, 'yb_top': yb_top, 'xb_bottom': xb_bottom, 'yb_bottom': yb_bottom, 'xb_left': xb_left, 'yb_left': yb_left, 'xb_right': xb_right, 'yb_right': yb_right }
ovals = {'ub': u, 'vb': v, 'pb': p, 'kb' : k, 'epsb': eps}
parameters = {'Re': 180., 'alpha': 1, 'beta': 1, 'gamma': 1, 'lambda': 0.1}
initializer = GlorotUniform(seed = 1234)

#model parameters
input1 = layers.Input(shape=(1,), name='x_input')
input2 = layers.Input(shape=(1,), name='y_input')
x = layers.Concatenate()([input1, input2])
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)

u_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)

v_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)

k_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)

e_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)

x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_1')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_2')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_3')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_4')(x)

u_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_u')(x)
v_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_v')(x)
k_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_k')(x)
e_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_eps')(x)

ou = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_u')(u_layer)
ov = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_v')(v_layer)
ok = layers.Dense(units=1,  activation='relu', use_bias=False, kernel_initializer=initializer, name='output_k')(k_layer)
oeps = layers.Dense(units=1,  activation='relu', use_bias=False, kernel_initializer=initializer, name='output_eps')(e_layer)

model = keras.Model([input1, input2], [ou, ov, ok, oeps])

model.summary()

# Training the model
loss_fn = keras.losses.MeanSquaredError()

initial_learning_rate = 1e-3

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=initial_learning_rate,
     decay_steps=10000,
     decay_rate=0.9,
     staircase=False)
optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule, beta_1 = 0.2, beta_2 = 0.7)

model_dict = {"nn_model": model}

metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='bound_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "mlx_loss": keras.metrics.Mean(name='fx_loss'),
           "mly_loss": keras.metrics.Mean(name='fy_loss'),
           "div_loss": keras.metrics.Mean(name='div_loss'),
           "Pe_loss": keras.metrics.Mean(name='Pe_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss'),
           "p_loss": keras.metrics.Mean(name='p_loss'),
           "k_loss": keras.metrics.Mean(name='k_loss'),
           "eps_loss": keras.metrics.Mean(name='eps_loss')
           }

cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, ibc_layer=False, mask=None)

log_dir = 'log_output/'
history = cm.run(epochs=1000, proj_name="k_eps_2D_channel_flow", log_dir=log_dir,
                 wb=False, verbose_freq=1000)

# Evaluation
cm.nn_model.save('Saved_model/', save_format='tf')
