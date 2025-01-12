import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers

import numpy as np
import time

tf.random.set_seed(1234)


def get_ibc_and_inner_data(nu, u_tau):

    # Boundary Points
    x_data = tf.linspace(0,200,100)
    y_data = tf.linspace(0.001 + 100*nu/u_tau,100,250)

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
    u_bottom = np.ones((X.shape[1],1))*((u_tau*tf.math.log(yb_bottom[0][0]/0.001)/0.418))
    u_left = ((u_tau*tf.math.log(yb_left/0.001)/0.418))

    v_bottom = np.zeros((X.shape[1],1))

    p_right = np.zeros((Y.shape[0]-2,1))

    k_bottom = np.ones((X.shape[1],1))*u_tau*u_tau/tf.sqrt(0.033)
    k_left = np.ones_like(u_left)*u_tau*u_tau/tf.sqrt(0.033)

    eps_bottom = np.ones((X.shape[1],1))*(u_tau**3)/(0.418*yb_bottom[0][0])
    eps_left = (u_tau**3)/(0.418*yb_left)

    x_train = grid_loc

    u_ob = np.vstack([u_bottom,u_left])
    v_ob = np.vstack([v_bottom])
    p_ob = np.vstack([p_right])
    k_ob = np.vstack([k_bottom,k_left])
    eps_ob = np.vstack([eps_bottom,eps_left])

    xd = x_train[:, 0:1]
    yd = x_train[:, 1:2]

    return xb_top, yb_top, xb_bottom, yb_bottom, xb_left, yb_left, xb_right, yb_right, xd, yd, u_ob, v_ob, p_ob, k_ob, eps_ob

class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn,
                 optimizer, metrics, ibc_layer=False, mask=None):
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

    @tf.function
    def pde_residual(self, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([self.xin, self.yin])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([self.xin, self.yin])
                u, v, P, k, eps = self.nn_model([self.xin, self.yin], training=True)
           
                sigma_k     = 1.
                sigma_eps   = 1.3
                C_eps_1     = 1.22
                C_eps_2     = 1.92
                C_mu        = 0.033

                k_zero_correction = 1.0
                eps_zero_correction = 0.1

                nu = 1.48e-5       
                nu_tau = C_mu * k**2 / (eps + eps_zero_correction)

            # first order derivatives wrt x
            ux = inner_tape.gradient(u, self.xin)
            vx = inner_tape.gradient(v, self.xin)
            px = inner_tape.gradient(P, self.xin)
            kx = inner_tape.gradient(k, self.xin)
            epsx = inner_tape.gradient(eps, self.xin)

            # first order derivatives wrt y
            uy = inner_tape.gradient(u, self.yin)
            vy = inner_tape.gradient(v, self.yin)
            py = inner_tape.gradient(P, self.yin)
            ky = inner_tape.gradient(k, self.yin)
            epsy = inner_tape.gradient(eps, self.yin)

            fx_x = (nu + nu_tau) * ux
            fx_y = (nu + nu_tau) * uy
            fy_x = (nu + nu_tau) * vx
            fy_y = (nu + nu_tau) * vy

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
        fx = (u * ux + v * uy + px) - (fx_xx + fx_yy)
        fy = (u * vx + v * vy + py) - (fy_xx + fy_yy)

        k_xx = outer_tape.gradient(k_x, self.xin)
        k_yy = outer_tape.gradient(k_y, self.yin)
        eps_xx = outer_tape.gradient(eps_x, self.xin)
        eps_yy = outer_tape.gradient(eps_y, self.yin)


        # k-epsilon equations
        k_eqn = u * kx + v * ky - (k_xx + k_yy) - P_k + eps
        eps_eqn = (u * epsx + v * epsy - (eps_xx + eps_yy) 
                    - (C_eps_1 * P_k - C_eps_2 * eps) * (eps / (k + k_zero_correction)))

        # Pressure equation 
        Pe = tf.abs(px) + tf.abs(py)
        
        hom_fully_dev = tf.abs(ux)+tf.abs(vx)+tf.abs(kx) + tf.abs(epsx) +tf.abs(ky)

        return fx, fy, div_u, Pe, k_eqn, eps_eqn, hom_fully_dev

    @tf.function
    def train_step(self):

        with tf.GradientTape(persistent=True) as tape:
            u_bottom, v_bottom,_, k_bottom, eps_bottom = self.nn_model([self.xb_bottom, self.yb_bottom], training=True)
            
            with tf.GradientTape(persistent=True) as innertape:
                innertape.watch([self.xb_top, self.yb_top, self.xb_left, self.yb_left, self.xb_right, self.yb_right])
                u_top, v_top, p_top, k_top, eps_top = self.nn_model([self.xb_top, self.yb_top], training=True)
                u_left, v_left, _, k_left, eps_left = self.nn_model([self.xb_left, self.yb_left], training=True)

            #top boundary symmetric condition           
            du_top = innertape.gradient(u_top, self.yb_top)
            dv_top = innertape.gradient(v_top, self.yb_top)
            dp_top = innertape.gradient(p_top, self.yb_top)
            dk_top = innertape.gradient(k_top, self.yb_top)
            deps_top = innertape.gradient(eps_top, self.yb_top)

            du_loss = tf.reduce_mean(tf.square(du_top))
            dv_loss = tf.reduce_mean(tf.square(dv_top))
            dp_loss = tf.reduce_mean(tf.square(dp_top))
            dk_loss = tf.reduce_mean(tf.square(dk_top))      
            deps_loss = tf.reduce_mean(tf.square(deps_top))

            u_pred = tf.concat([u_bottom,u_left], axis=0)
            k_pred = tf.concat([k_bottom,k_left], axis=0)
            eps_pred = tf.concat([eps_bottom,eps_left], axis=0)

            #Dirichlet conditions at inlet and bottom boundary
            u_loss = self.loss_fn(self.ub, u_pred)
            v_loss = self.loss_fn(self.vb, v_bottom) + tf.reduce_mean(tf.square(v_left))
            k_loss = self.loss_fn(self.kb , k_pred)
            eps_loss = self.loss_fn(self.epsb , eps_pred)

            #boundary losses
            std_loss = u_loss + v_loss + du_loss + dv_loss + dp_loss + dk_loss + deps_loss + k_loss + eps_loss

            #pde losses
            fx, fy, div_u, Pe, keq, epseq, fully_dev= self.pde_residual(True)
            fx_loss = tf.reduce_mean(tf.square(fx))
            fy_loss = tf.reduce_mean(tf.square(fy))
            div_loss = tf.reduce_mean(tf.square(div_u))
            Pe_loss = tf.reduce_mean(tf.square(Pe))
            keq_loss = tf.reduce_mean(tf.square(keq))
            epseq_loss = tf.reduce_mean(tf.square(epseq))
            fully_dev_loss = tf.reduce_mean((fully_dev))
            
            #total pde losses with weights
            residual_loss = 10*(fx_loss  + fy_loss) +  100*div_loss + 1000*fully_dev_loss + 100*Pe_loss + 10*(keq_loss) + (epseq_loss) 
            
            #total loss
            loss = 100*std_loss + residual_loss

        #gradient and optimization
        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))
        grad_std = tape.gradient(std_loss, self.nn_model.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
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

    def run(self, epochs, log_dir, wb=False, verbose_freq=1000, grad_freq=1000):

        self.reset_metrics()
        history = {"loss": [], "std_loss": [], "residual_loss": [], "mom-x_loss": [], "mom-y_loss": [], "div_loss": [], "k_loss": [], "eps_loss": [],
                   "mag_res_grads": [],"mag_bound_grads": []}
                
        start_time = time.time()

        for epoch in range(epochs):
            
            logs, grads, grad_std, grad_res = self.train_step()
            if (epoch+1) % grad_freq == 0:
                default_grad = tf.zeros([1])
                grad_std = [grad if grad is not None else default_grad for grad in grad_std]
                grad_res = [grad if grad is not None else default_grad for grad in grad_res]
                grads = [grad if grad is not None else default_grad for grad in grads]

                mag_res_grads = tf.sqrt(tf.reduce_sum([tf.reduce_sum(grd ** 2) for grd in grad_res])).numpy()
                mag_bound_grads = tf.sqrt(tf.reduce_sum([tf.reduce_sum(grd_1 ** 2) for grd_1 in grad_std])).numpy()
            
            tae = time.time() - start_time
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                history['mag_res_grads'].append(mag_res_grads)
                history['mag_bound_grads'].append(mag_bound_grads)
                print(f"Time: {tae / 60:.4f}min")


        return history

