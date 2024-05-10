import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers

import numpy as np
from pyDOE import lhs
import time

tf.random.set_seed(1234)


def get_ibc_and_inner_data(num_samples, nu):

    # Boundary Points
    x_data = tf.linspace(0,1,11)
    y_data = tf.linspace(63*nu,1,200)

    X, Y = np.meshgrid(x_data,y_data)
    grid_loc = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))           

    lb = grid_loc.min(0)
    ub = grid_loc.max(0)
    
    x_bottom = np.hstack((X[0:1,:].T, Y[0:1,:].T))
    x_top = np.hstack((X[-1:,:].T, Y[-1:,:].T))
    x_left = np.hstack((X[1:-1,0:1], Y[1:-1,0:1]))
    x_right = np.hstack((X[1:-1,-1:], Y[1:-1,-1:]))

    xb_top = x_top[:, 0:1]
    yb_top = x_top[:, 1:2]

    xb_bottom = x_bottom[:, 0:1]
    yb_bottom = x_bottom[:, 1:2]

    xb_left = x_left[:, 0:1]
    yb_left = x_left[:, 1:2]

    xb_right = x_right[:, 0:1]
    yb_right = x_right[:, 1:2]
    
    # boundary conditions
    u_bottom = np.ones((X.shape[1],1))*((tf.math.log(9.793*yb_bottom[0][0]/nu)/0.418))
    v_bottom = np.zeros((X.shape[1],1))

    p_right = np.zeros((Y.shape[0]-2,1))

    k_bottom = np.ones((X.shape[1],1))/tf.sqrt(0.09)

    eps_bottom = np.ones((X.shape[1],1))/(0.418*yb_bottom[0][0])

    X_f_train = lb + (ub-lb)*lhs(2, num_samples)

    x_train = np.vstack((x_bottom, x_top, x_left, x_right, X_f_train))
 
    u_ob = np.vstack([u_bottom])
    v_ob = np.vstack([v_bottom])
    p_ob = np.vstack([p_right])
    k_ob = np.vstack([k_bottom])
    eps_ob = np.vstack([eps_bottom])

    xd = x_train[:, 0:1]
    yd = x_train[:, 1:2]

    return xb_top, yb_top, xb_bottom, yb_bottom, xb_left, yb_left, xb_right, yb_right, xd, yd, u_ob, v_ob, p_ob, k_ob, eps_ob

class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn,
                 optimizer, metrics, parameters, ibc_layer=False, mask=None):
        self.nn_model = get_models['nn_model']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ibc_layer = ibc_layer

        self.xin = tf.constant(inputs['xin'], dtype=tf.float32)
        self.yin = tf.constant(inputs['yin'], dtype=tf.float32)
        self.xb_top = tf.constant(inputs['xb_top'], dtype=tf.float32)
        self.yb_top = tf.constant(inputs['yb_top'], dtype=tf.float32)
        self.xb_bottom = tf.constant(inputs['xb_bottom'], dtype=tf.float32)
        self.yb_bottom = tf.constant(inputs['yb_bottom'], dtype=tf.float32)
        self.xb_left = tf.constant(inputs['xb_left'], dtype=tf.float32)
        self.yb_left = tf.constant(inputs['yb_left'], dtype=tf.float32)
        self.xb_right = tf.constant(inputs['xb_right'], dtype=tf.float32)
        self.yb_right = tf.constant(inputs['yb_right'], dtype=tf.float32)
        
        self.xb = tf.concat([self.xb_top],0)
        self.yb = tf.concat([self.xb_top],0)

        self.ub = tf.constant(outputs['ub'], dtype=tf.float32)
        self.vb = tf.constant(outputs['vb'], dtype=tf.float32)
        self.pb = tf.constant(outputs['pb'], dtype=tf.float32)
        self.kb = tf.constant(outputs['kb'], dtype=tf.float32)
        self.epsb = tf.constant(outputs['epsb'], dtype=tf.float32)

        self.loss_tracker = metrics['loss']
        self.std_loss_tracker = metrics['boundary_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.mlx_loss_tracker = metrics['mlx_loss']
        self.mly_loss_tracker = metrics['mly_loss']
        self.div_loss_tracker = metrics['div_loss']
        self.Pe_loss_tracker = metrics['Pe_loss']
        self.k_loss_tracker = metrics['k_loss']
        self.eps_loss_tracker = metrics['eps_loss']

        self.Re = tf.constant(parameters['Re'], dtype=tf.float32)

        self.Re_no = self.Re

    @tf.function
    def pde_residual(self, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([self.xin, self.yin])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([self.xin, self.yin])
                u, v, k, eps = self.nn_model([self.xin, self.yin], training=True)

                sigma_k     = 1.
                sigma_eps   = 1.3
                C_eps_1     = 1.44
                C_eps_2     = 1.92
                C_mu        = 0.09

                k_zero_correction = .01
                eps_zero_correction = 0.01
                Re = self.Re
                nu = 1/Re
        
                nu_tau = C_mu * k**2 / (eps + eps_zero_correction)

            # first order derivatives wrt x
            ux = inner_tape.gradient(u, self.xin)
            vx = inner_tape.gradient(v, self.xin)
            kx = inner_tape.gradient(k, self.xin)
            epsx = inner_tape.gradient(eps, self.xin)


            uy = inner_tape.gradient(u, self.yin)
            vy = inner_tape.gradient(v, self.yin)
            ky = inner_tape.gradient(k, self.yin)
            epsy = inner_tape.gradient(eps, self.yin)

            fx_x = (nu + nu_tau) * (ux + ux)
            fx_y = (nu + nu_tau) * (uy + vx)
            fy_x = (nu + nu_tau) * (vx + uy)
            fy_y = (nu + nu_tau) * (vy + vy)

            k_x = (nu + nu_tau/sigma_k) * kx
            k_y = (nu + nu_tau/sigma_k) * ky
            eps_x = (nu + nu_tau/sigma_eps) * epsx
            eps_y = (nu + nu_tau/sigma_eps) * epsy

        P_k = nu_tau * ( 2 * ux ** 2 + 2 * vy ** 2 + uy ** 2 + vx ** 2 + 2 * uy * vx)

        # Continuity equation
        div_u = ux + vy

        fx_xx = outer_tape.gradient(fx_x, self.xin)
        fx_yy = outer_tape.gradient(fx_y, self.yin)
        fy_xx = outer_tape.gradient(fy_x, self.xin)
        fy_yy = outer_tape.gradient(fy_y, self.yin)

        # Momentum equation calculation
        fx = (u * ux + v * uy -1) - fx_xx - fx_yy
        fy = (u * vx + v * vy) - fy_xx - fy_yy

        k_xx = outer_tape.gradient(k_x, self.xin)
        k_yy = outer_tape.gradient(k_y, self.yin)
        eps_xx = outer_tape.gradient(eps_x, self.xin)
        eps_yy = outer_tape.gradient(eps_y, self.yin)

        # k-epsilon equations
        k_eqn = u * kx + v * ky - (k_xx + k_yy) - P_k + eps
        eps_eqn = (u * epsx + v * epsy - (eps_xx + eps_yy) 
                    - (C_eps_1 * P_k - C_eps_2 * eps) * (eps / (k + k_zero_correction)))

        fully_dev = tf.abs(ux)+tf.abs(vx)+tf.abs(kx) + tf.abs(epsx)

        return fx, fy, div_u, k_eqn, eps_eqn, fully_dev

    @tf.function
    def train_step(self):

        with tf.GradientTape(persistent=True) as tape:
            u_pred, v_pred, k_pred, eps_pred = self.nn_model([self.xb_bottom, self.yb_bottom], training=True)
            
            with tf.GradientTape(persistent=True) as innertape:
                innertape.watch([self.xb_top, self.yb_top, self.xb_left, self.yb_left, self.xb_right, self.yb_right])
                u_top, v_top, k_top, eps_top = self.nn_model([self.xb_top, self.yb_top], training=True)
                u_left, v_left, k_left, eps_left = self.nn_model([self.xb_left, self.yb_left], training=True)
                u_right, v_right, k_right, eps_right = self.nn_model([self.xb_right, self.yb_right], training=True)
                
            
            du_top = innertape.gradient(u_top, self.yb_top)
            dk_top = innertape.gradient(k_top, self.yb_top)
            deps_top = innertape.gradient(eps_top, self.yb_top)
            
            u_loss = self.loss_fn(self.ub, u_pred)
            v_loss = self.loss_fn(self.vb, v_pred) + tf.reduce_mean(tf.square(v_top))
            k_loss = self.loss_fn(self.kb , k_pred)
            eps_loss = self.loss_fn(self.epsb , eps_pred)


            du_loss = tf.reduce_mean(tf.square(du_top))
            dk_loss = tf.reduce_mean(tf.square(dk_top))      
            deps_loss = tf.reduce_mean(tf.square(deps_top))

            std_loss = u_loss + v_loss + du_loss  + dk_loss + deps_loss + eps_loss + k_loss # + dv_loss
            
            fx, fy, div_u, keq, epseq, fully_dev= self.pde_residual(True)
            fx_loss = tf.reduce_mean(tf.square(fx))
            fy_loss = tf.reduce_mean(tf.square(fy))
            div_loss = tf.reduce_mean(tf.square(div_u))
            keq_loss = tf.reduce_mean(tf.square(keq))
            epseq_loss = tf.reduce_mean(tf.square(epseq))
            fully_dev_loss = tf.reduce_mean((fully_dev))

            residual_loss = 10*(fx_loss  + fy_loss) +  100*div_loss + 1000*fully_dev_loss + 1*(keq_loss) + (epseq_loss) # + Pey_loss
            loss = 100*std_loss + residual_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))
        grad_std = tape.gradient(std_loss, self.nn_model.trainable_weights)
        grad_res = tape.gradient(residual_loss, self.nn_model.trainable_weights)

        self.loss_tracker.update_state(loss)
        self.std_loss_tracker.update_state(std_loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.mlx_loss_tracker.update_state(fx_loss)
        self.mly_loss_tracker.update_state(fy_loss)
        self.div_loss_tracker.update_state(div_loss)
        self.k_loss_tracker.update_state(keq_loss)
        self.eps_loss_tracker.update_state(epseq_loss)

        return {"loss": self.loss_tracker.result(), "std_loss": self.std_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result(), 'mom-x_loss': self.mlx_loss_tracker.result(),
                "mom-y_loss": self.mly_loss_tracker.result(), "div_loss": self.div_loss_tracker.result(),
                "k_loss": self.k_loss_tracker.result(), "eps_loss": self.eps_loss_tracker.result()}, grads, \
               grad_std, grad_res

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.std_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.mlx_loss_tracker.reset_state()
        self.mly_loss_tracker.reset_state()
        self.div_loss_tracker.reset_state()
        self.Pe_loss_tracker.reset_state()
        self.k_loss_tracker.reset_state()
        self.eps_loss_tracker.reset_state()

    def run(self, epochs, proj_name, log_dir,wb=False, verbose_freq=1000, grad_freq=5000):

        self.reset_metrics()
        history = {"loss": [], "std_loss": [], "residual_loss": [], "mom-x_loss": [], "mom-y_loss": [], "div_loss": [], "k_loss": [], "eps_loss": []}

        start_time = time.time()

        for epoch in range(epochs):
            
            logs, grads, grad_std, grad_res = self.train_step()
            if (epoch+1) % grad_freq == 0:
                default_grad = tf.zeros([1])
                grad_std = [grad if grad is not None else default_grad for grad in grad_std]
                grad_res = [grad if grad is not None else default_grad for grad in grad_res]
                grads = [grad if grad is not None else default_grad for grad in grads]

            tae = time.time() - start_time
            if (epoch + 1) % verbose_freq == 0 or epoch==100:
                print(f'''Epoch:{epoch + 1}/{epochs} for Re {self.Re_no}''')
                for key, value in logs.items():
                    history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")

        return history