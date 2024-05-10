import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np

from keras.initializers import GlorotUniform

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.chdir('./')

from Sexbierum_wake_model import PdeModel, get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
nu = 1.48e-5   #kinematic viscosity
u_ref=10.      #reference velocity
z_ref=35.      #reference height
TI_ref=0.1     #reference turbulence intensity
Radius=15.05   #radius of turbine
C_T=0.75       #thrust coefficient

#boundary and grid data
#inlet, outlet, bottom, top, front, back - 6 boundary coordinates
#x_train - domain coordinates
#x_disk - actuator disc coordinates
#inlet_uvwke, outlet_p, bottom_uvwke, top_uvwke, front_uvwke, back_uvwke - {boundary}_{boundary_values of different variables}
inlet, outlet, bottom, top, front, back, x_train, x_disk, inlet_uvwke, outlet_p, bottom_uvwke, top_uvwke, front_uvwke, back_uvwke = get_ibc_and_inner_data(nu=nu, u_ref=u_ref, z_ref=z_ref, TI_ref=TI_ref, Radius=Radius)

#inputs and outputs
ivals = {'xin': x_train[:,0:1], 'yin': x_train[:,1:2], 'zin': x_train[:,2:3],
         'xb_inlet' : inlet[:,0:1],     'yb_inlet' : inlet[:,1:2],  'zb_inlet' : inlet[:,2:3],  
         'xb_outlet': outlet[:,0:1],    'yb_outlet': outlet[:,1:2], 'zb_outlet': outlet[:,2:3],  
         'xb_bottom': bottom[:,0:1],    'yb_bottom': bottom[:,1:2], 'zb_bottom': bottom[:,2:3],  
         'xb_top'   : top[:,0:1],       'yb_top'   : top[:,1:2],    'zb_top'   : top[:,2:3],  
         'xb_front' : front[:,0:1],     'yb_front' : front[:,1:2],  'zb_front' : front[:,2:3],  
         'xb_back'  : back[:,0:1],      'yb_back'  : back[:,1:2],   'zb_back'  : back[:,2:3],  
         'xd_disk'  : x_disk[:,0:1],    'yd_disk'  : x_disk[:,1:2], 'zd_disk'  : x_disk[:,2:3] }
ovals = {'ub_inlet': inlet_uvwke[:,0:1], 'ub_bottom': bottom_uvwke[:,0:1], 'ub_top': top_uvwke[:,0:1], 'ub_front': front_uvwke[:,0:1], 'ub_back': back_uvwke[:,0:1],
           'vb_inlet': inlet_uvwke[:,1:2], 'vb_bottom': bottom_uvwke[:,1:2], 'vb_top': top_uvwke[:,1:2], 'vb_front': front_uvwke[:,1:2], 'vb_back': back_uvwke[:,1:2],
           'wb_inlet': inlet_uvwke[:,2:3], 'wb_bottom': bottom_uvwke[:,2:3], 'wb_top': top_uvwke[:,2:3], 'wb_front': front_uvwke[:,2:3], 'wb_back': back_uvwke[:,2:3],
           'pb': outlet_p[:,0:1], 
           'kb_inlet' : inlet_uvwke[:,3:4], 'kb_bottom' : bottom_uvwke[:,3:4], 'kb_top': top_uvwke[:,3:4], 'kb_front': front_uvwke[:,3:4], 'kb_back': back_uvwke[:,3:4],
           'epsb_inlet' : inlet_uvwke[:,4:5], 'epsb_bottom': bottom_uvwke[:,4:5], 'epsb_top' : top_uvwke[:,4:5], 'epsb_front': front_uvwke[:,4:5], 'epsb_back': back_uvwke[:,4:5]}
parameters = {'C_T': C_T, 'u_ref': u_ref}
initializer = GlorotUniform(seed = 1234)

model = tf.keras.models.load_model('Base_model')

model.summary()

# Training the model
loss_fn = keras.losses.MeanSquaredError()

initial_learning_rate = 1e-4*0.9**5

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=initial_learning_rate,
     decay_steps=5000,
     decay_rate=0.9,
     staircase=False)
optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule, beta_1=0.7, beta_2=0.7)

model_dict = {"nn_model": model}

metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='bound_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "mlx_loss": keras.metrics.Mean(name='fx_loss'),
           "mly_loss": keras.metrics.Mean(name='fy_loss'),
           "mlz_loss": keras.metrics.Mean(name='fz_loss'),
           "div_loss": keras.metrics.Mean(name='div_loss'),
           "Pe_loss": keras.metrics.Mean(name='Pe_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss'),
           "p_loss": keras.metrics.Mean(name='p_loss'),
           "k_loss": keras.metrics.Mean(name='k_loss'),
           "eps_loss": keras.metrics.Mean(name='eps_loss')
           }

cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=12, ibc_layer=False, mask=None)

log_dir = 'log_output/'
history = cm.run(epochs=5000, proj_name="Actuator_disc_3D_TL", log_dir=log_dir,
                 wb=False, verbose_freq=1000, plot_freq=1000)

# Evaluation
cm.nn_model.save('Transfer_learned_model/', save_format='tf')
