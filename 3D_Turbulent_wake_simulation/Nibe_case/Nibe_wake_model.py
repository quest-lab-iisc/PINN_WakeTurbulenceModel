import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers

import numpy as np
import time
from idaes.core.surrogate.pysmo.sampling import HammersleySampling

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
tf.random.set_seed(1234)

def get_ibc_and_inner_data(nu, u_ref, z_ref, TI_ref, Radius):

    kappa = 0.418
    c_mu = 0.033

    k_ref = 1.5 * (u_ref*TI_ref)**2
    u_tau = np.sqrt(k_ref * np.sqrt(c_mu))
    z_0 = z_ref/(np.exp(u_ref*kappa/u_tau)-1)

    x_dom = np.linspace(-10 * Radius, 25 * Radius,36*2-1)
    y_dom = np.linspace(-10 * Radius, 10 * Radius,21*2-1)
    z_dom = np.linspace(0.02* Radius ,15 * Radius,16*3-2)

    X,Y,Z = np.meshgrid(x_dom,y_dom,z_dom)

    lb_refined = [-6.*Radius, -2*Radius, -2*Radius+z_ref]
    ub_refined = [6.*Radius, 2*Radius, 2*Radius+z_ref]

    refined_loc = HammersleySampling([lb_refined, ub_refined], sampling_type='creation', number_of_samples=4000).sample_points()
    R_refined = np.linalg.norm((refined_loc[:,1:2],refined_loc[:,2:3]-z_ref),axis=0)
    indices_refined = np.where((R_refined > 2*Radius) | ((R_refined<=Radius)&(np.abs(refined_loc[:,0:1])<=0.5*Radius)))
    refined_loc[indices_refined[0]]=np.nan
    refined_loc = (refined_loc[np.logical_not(np.isnan(refined_loc))]).reshape(-1,3)  

    lb_disk = [-0.5*Radius, -Radius, -Radius+z_ref]
    ub_disk = [0.5*Radius, Radius, Radius+z_ref]

    x_disk_loc = HammersleySampling([lb_disk, ub_disk], sampling_type='creation', number_of_samples=4000).sample_points()
    R_disk = np.linalg.norm((x_disk_loc[:,1:2],x_disk_loc[:,2:3]-z_ref),axis=0)
    indices_disk = np.where((R_disk > Radius))
    x_disk_loc[indices_disk[0]]=np.nan
    x_disk_loc = (x_disk_loc[np.logical_not(np.isnan(x_disk_loc))]).reshape(-1,3)


    inlet = np.hstack((X[:,0:1,:].flatten()[:,None], Y[:,0:1,:].flatten()[:,None], Z[:,0:1,:].flatten()[:,None]))
    outlet = np.hstack((X[:,-1,:].flatten()[:,None], Y[:,-1,:].flatten()[:,None], Z[:,-1,:].flatten()[:,None]))
    bottom= np.hstack((X[:,:,0:1].flatten()[:,None], Y[:,:,0:1].flatten()[:,None], Z[:,:,0:1].flatten()[:,None]))
    top = np.hstack((X[:,:,-1].flatten()[:,None], Y[:,:,-1].flatten()[:,None], Z[:,:,-1].flatten()[:,None]))
    front = np.hstack((X[0:1,:,:].flatten()[:,None], Y[0:1,:,:].flatten()[:,None], Z[0:1,:,:].flatten()[:,None]))
    back = np.hstack((X[-1,:,:].flatten()[:,None], Y[-1,:,:].flatten()[:,None], Z[-1,:,:].flatten()[:,None]))

    grid_loc = np.hstack((X.flatten()[:,None], Y.flatten()[:,None],Z.flatten()[:,None]))

    grid_loc = np.hstack((X.flatten()[:,None], Y.flatten()[:,None],Z.flatten()[:,None]))     

    lb = grid_loc.min(0).tolist()
    ub = grid_loc.max(0).tolist()

    train_domain = HammersleySampling([lb, ub], sampling_type='creation', number_of_samples=40000).sample_points()
    RR = np.linalg.norm((train_domain[:,1:2],train_domain[:,2:3]-z_ref),axis=0)
    indices_domain = np.where((RR<= 2*Radius) & (np.abs(train_domain[:,0:1])<6*Radius))
    train_domain[indices_domain[0]]=np.nan
    train_domain = (train_domain[np.logical_not(np.isnan(train_domain))]).reshape(-1,3)

    x_train = np.vstack((train_domain,refined_loc,inlet,outlet,bottom,top,front,back))         
    
    u_inlet = (u_tau*tf.math.log((inlet[:,2:3] + z_0)/z_0)/kappa)
    v_inlet = np.zeros_like(u_inlet)
    w_inlet = np.zeros_like(u_inlet)
    u_bottom = (u_tau*tf.math.log((bottom[:,2:3] + z_0)/z_0)/kappa)
    v_bottom = np.zeros_like(u_bottom)
    w_bottom = np.zeros_like(u_bottom)
    u_top = (u_tau*tf.math.log((top[:,2:3] + z_0)/z_0)/kappa)
    v_top = np.zeros_like(u_top)
    w_top = np.zeros_like(u_top)
    u_front = (u_tau*tf.math.log((front[:,2:3] + z_0)/z_0 )/kappa)
    v_front = np.zeros_like(u_front)
    w_front = np.zeros_like(u_front)
    u_back = (u_tau*tf.math.log((back[:,2:3] + z_0)/z_0 )/kappa)
    v_back = np.zeros_like(u_back)
    w_back = np.zeros_like(u_back)

    #pressure boundary
    p_outlet = np.zeros_like(outlet[:,0:1])

    #k boundary
    k_inlet = np.ones_like(u_inlet)*u_tau*u_tau/tf.sqrt(c_mu)
    k_bottom = np.ones_like(bottom[:,0:1])*u_tau*u_tau/tf.sqrt(c_mu)
    k_top = np.ones_like(top[:,0:1])*u_tau*u_tau/tf.sqrt(c_mu)
    k_front = np.ones_like(front[:,0:1])*u_tau*u_tau/tf.sqrt(c_mu)
    k_back = np.ones_like(front[:,0:1])*u_tau*u_tau/tf.sqrt(c_mu)

    #eps boundary
    eps_inlet = (u_tau**3)/(kappa*(inlet[:,2:3] + z_0 ))
    eps_bottom = (u_tau**3)/(kappa*(bottom[:,2:3] + z_0 ))
    eps_top = (u_tau**3)/(kappa*(top[:,2:3] + z_0 ))
    eps_front = (u_tau**3)/(kappa*(front[:,2:3] + z_0  ))
    eps_back = (u_tau**3)/(kappa*(back[:,2:3] + z_0  ))

    inlet_uvwke = np.hstack([u_inlet, v_inlet, w_inlet, k_inlet, eps_inlet])
    outlet_p = np.hstack([p_outlet])
    bottom_uvwke = np.hstack([u_bottom, v_bottom, w_bottom, k_bottom, eps_bottom])
    top_uvwke = np.hstack([u_top, v_top, w_top, k_top, eps_top])
    front_uvwke = np.hstack([u_front, v_front, w_front, k_front, eps_front])
    back_uvwke = np.hstack([u_back, v_back, w_back, k_back, eps_back])

    return inlet, outlet, bottom, top, front, back, x_train, x_disk_loc, inlet_uvwke, outlet_p, bottom_uvwke, top_uvwke, front_uvwke, back_uvwke

class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn,
                 optimizer, metrics, parameters, batches=1, ibc_layer=False, mask=None):
        self.nn_model = get_models['nn_model']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ibc_layer = ibc_layer
        self.batches = batches

        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['yin'], inputs['zin'], batch=batches).cache()
        self.disk_data = self.create_data_pipeline(inputs['xd_disk'], inputs['yd_disk'], inputs['zd_disk'], batch=batches).cache()
        self.inlet_data = self.create_data_pipeline(
            inputs['xb_inlet'], inputs['yb_inlet'],inputs['zb_inlet'], 
            outputs['ub_inlet'], outputs['vb_inlet'], outputs['wb_inlet'],
            outputs['kb_inlet'],outputs['epsb_inlet'], batch=batches).cache()
        self.outlet_data = self.create_data_pipeline(
            inputs['xb_outlet'], inputs['yb_outlet'],inputs['zb_outlet'],
            outputs['pb'], batch=batches).cache()
        self.bottom_data = self.create_data_pipeline(
            inputs['xb_bottom'], inputs['yb_bottom'],inputs['zb_bottom'], 
            outputs['ub_bottom'], outputs['vb_bottom'], outputs['wb_bottom'],
            outputs['kb_bottom'], outputs['epsb_bottom'], batch=batches).cache()
        self.top_data = self.create_data_pipeline(
            inputs['xb_top'], inputs['yb_top'],inputs['zb_top'], 
            outputs['ub_top'], outputs['vb_top'], outputs['wb_top'],
            outputs['kb_top'], outputs['epsb_top'], batch=batches).cache()
        self.front_data = self.create_data_pipeline(
            inputs['xb_front'], inputs['yb_front'],inputs['zb_front'],
            outputs['ub_front'], outputs['vb_front'], outputs['wb_front'],
            outputs['kb_front'],outputs['epsb_front'], batch=batches).cache()
        self.back_data = self.create_data_pipeline(
            inputs['xb_back'], inputs['yb_back'],inputs['zb_back'],
            outputs['ub_back'], outputs['vb_back'], outputs['wb_back'],
            outputs['kb_back'],outputs['epsb_back'], batch=batches).cache()
        
        self.C_T = tf.constant(parameters['C_T'], dtype=tf.float32)
        self.u_ref = tf.constant(parameters['u_ref'], dtype=tf.float32)

        self.loss_tracker = metrics['loss']
        self.std_loss_tracker = metrics['boundary_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.mlx_loss_tracker = metrics['mlx_loss']
        self.mly_loss_tracker = metrics['mly_loss']
        self.mlz_loss_tracker = metrics['mlz_loss']
        self.div_loss_tracker = metrics['div_loss']
        self.Pe_loss_tracker = metrics['Pe_loss']
        self.k_loss_tracker = metrics['k_loss']
        self.eps_loss_tracker = metrics['eps_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0]) / batch))
        return dataset
    
    @tf.function
    def pde_residual(self, xin, yin, zin, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([xin, yin, zin])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([xin, yin, zin])
                u, v, w, P, k, eps = self.nn_model([xin, yin, zin], training=True)

                sigma_k     = 1.
                sigma_eps   = 1.3
                C_eps_1     = 1.176
                C_eps_2     = 1.92
                C_eps_4     = 0.37
                C_mu        = 0.033

                k_zero_correction = .01
                eps_zero_correction = .01
                nu = 1.48e-5
        
                nu_tau = C_mu * k**2 / (eps + eps_zero_correction)

            # first order derivatives wrt x
            ux = inner_tape.gradient(u, xin)
            vx = inner_tape.gradient(v, xin)
            wx = inner_tape.gradient(w, xin)
            px = inner_tape.gradient(P, xin)
            kx = inner_tape.gradient(k, xin)
            epsx = inner_tape.gradient(eps, xin)

            # first order derivatives wrt y
            uy = inner_tape.gradient(u, yin)
            vy = inner_tape.gradient(v, yin)
            wy = inner_tape.gradient(w, yin)
            py = inner_tape.gradient(P, yin)
            ky = inner_tape.gradient(k, yin)
            epsy = inner_tape.gradient(eps, yin)
        
            # # first order derivatives wrt z
            uz = inner_tape.gradient(u, zin)
            vz = inner_tape.gradient(v, zin)
            wz = inner_tape.gradient(w, zin)
            pz = inner_tape.gradient(P, zin)
            kz = inner_tape.gradient(k, zin)
            epsz = inner_tape.gradient(eps, zin)

            fx_x = (nu + nu_tau) * (ux + ux)
            fx_y = (nu + nu_tau) * (uy + vx)
            fx_z = (nu + nu_tau) * (uz + wx)
            fy_x = (nu + nu_tau) * (vx + uy)
            fy_y = (nu + nu_tau) * (vy + vy)
            fy_z = (nu + nu_tau) * (vz + wy)
            fz_x = (nu + nu_tau) * (wx + uz)
            fz_y = (nu + nu_tau) * (wy + vz)
            fz_z = (nu + nu_tau) * (wz + wz)

            k_x = (nu + nu_tau/sigma_k) * kx
            k_y = (nu + nu_tau/sigma_k) * ky
            k_z = (nu + nu_tau/sigma_k) * kz
            eps_x = (nu + nu_tau/sigma_eps) * epsx
            eps_y = (nu + nu_tau/sigma_eps) * epsy
            eps_z = (nu + nu_tau/sigma_eps) * epsz

        P_k = nu_tau * ( 2 * ux ** 2 + 2 * vy ** 2 + 2 * wz ** 2 + (uy + vx)**2 + (uz + wx)**2 + (vz + wy)**2)
    
        
        # Continuity equation
        div_u = ux + vy + wz

        fx_xx = outer_tape.gradient(fx_x, xin)
        fx_yy = outer_tape.gradient(fx_y, yin)
        fx_zz = outer_tape.gradient(fx_z, zin)
        fy_xx = outer_tape.gradient(fy_x, xin)
        fy_yy = outer_tape.gradient(fy_y, yin)
        fy_zz = outer_tape.gradient(fy_z, zin)
        fz_xx = outer_tape.gradient(fz_x, xin)
        fz_yy = outer_tape.gradient(fz_y, yin)
        fz_zz = outer_tape.gradient(fz_z, zin)

        # Momentum equation calculation
        fx = (u * ux + v * uy + w * uz + px) - (fx_xx + fx_yy + fx_zz)
        fy = (u * vx + v * vy + w * vz + py) - (fy_xx + fy_yy + fy_zz)
        fz = (u * wx + v * wy + w * wz + pz) - (fz_xx + fz_yy + fz_zz)

        k_xx = outer_tape.gradient(k_x, xin)
        k_yy = outer_tape.gradient(k_y, yin)
        k_zz = outer_tape.gradient(k_z, zin)
        eps_xx = outer_tape.gradient(eps_x, xin)
        eps_yy = outer_tape.gradient(eps_y, yin)
        eps_zz = outer_tape.gradient(eps_z, zin)

        # k-epsilon equations
        k_eqn = u * kx + v * ky + w * kz - (k_xx + k_yy + k_zz) - P_k + eps
        eps_eqn = (u * epsx + v * epsy + w * epsz - (eps_xx + eps_yy + eps_zz) 
                    - (C_eps_1 * P_k - C_eps_2 * eps) * (eps / (k + k_zero_correction)))

        return fx, fy, fz, div_u, k_eqn, eps_eqn

    @tf.function
    def act_disc_pde_residual(self, x_disk, y_disk, z_disk, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([x_disk, y_disk, z_disk])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([x_disk, y_disk, z_disk])
                u, v, w, P, k, eps = self.nn_model([x_disk, y_disk, z_disk], training=True)

                sigma_k     = 1.
                sigma_eps   = 1.3
                C_eps_1     = 1.176
                C_eps_2     = 1.92
                C_eps_4     = 0.37
                C_mu        = 0.033

                k_zero_correction = .01
                eps_zero_correction = .01
                nu = 1.48e-5
        
                nu_tau = C_mu * k**2 / (eps + eps_zero_correction)

            ux = inner_tape.gradient(u, x_disk)
            vx = inner_tape.gradient(v, x_disk)
            wx = inner_tape.gradient(w, x_disk)
            px = inner_tape.gradient(P, x_disk)
            kx = inner_tape.gradient(k, x_disk)
            epsx = inner_tape.gradient(eps, x_disk)

            # first order derivatives wrt y
            uy = inner_tape.gradient(u, y_disk)
            vy = inner_tape.gradient(v, y_disk)
            wy = inner_tape.gradient(w, y_disk)
            py = inner_tape.gradient(P, y_disk)
            ky = inner_tape.gradient(k, y_disk)
            epsy = inner_tape.gradient(eps, y_disk)
        
            # # first order derivatives wrt z
            uz = inner_tape.gradient(u, z_disk)
            vz = inner_tape.gradient(v, z_disk)
            wz = inner_tape.gradient(w, z_disk)
            pz = inner_tape.gradient(P, z_disk)
            kz = inner_tape.gradient(k, z_disk)
            epsz = inner_tape.gradient(eps, z_disk)

            fx_x = (nu + nu_tau) * (ux + ux)
            fx_y = (nu + nu_tau) * (uy + vx)
            fx_z = (nu + nu_tau) * (uz + wx)
            fy_x = (nu + nu_tau) * (vx + uy)
            fy_y = (nu + nu_tau) * (vy + vy)
            fy_z = (nu + nu_tau) * (vz + wy)
            fz_x = (nu + nu_tau) * (wx + uz)
            fz_y = (nu + nu_tau) * (wy + vz)
            fz_z = (nu + nu_tau) * (wz + wz)

            k_x = (nu + nu_tau/sigma_k) * kx
            k_y = (nu + nu_tau/sigma_k) * ky
            k_z = (nu + nu_tau/sigma_k) * kz
            eps_x = (nu + nu_tau/sigma_eps) * epsx
            eps_y = (nu + nu_tau/sigma_eps) * epsy
            eps_z = (nu + nu_tau/sigma_eps) * epsz

        P_k = nu_tau * ( 2 * ux ** 2 + 2 * vy ** 2 + 2 * wz ** 2 + (uy + vx)**2 + (uz + wx)**2 + (vz + wy)**2)
    
        # Continuity equation
        div_u = ux + vy + wz

        fx_xx = outer_tape.gradient(fx_x, x_disk)
        fx_yy = outer_tape.gradient(fx_y, y_disk)
        fx_zz = outer_tape.gradient(fx_z, z_disk)
        fy_xx = outer_tape.gradient(fy_x, x_disk)
        fy_yy = outer_tape.gradient(fy_y, y_disk)
        fy_zz = outer_tape.gradient(fy_z, z_disk)
        fz_xx = outer_tape.gradient(fz_x, x_disk)
        fz_yy = outer_tape.gradient(fz_y, y_disk)
        fz_zz = outer_tape.gradient(fz_z, z_disk)


        # Momentum equation calculation
        fx = (u * ux + v * uy + w * uz + px) - (fx_xx + fx_yy + fx_zz) + (self.C_T*0.5*((self.u_ref)**2)*tf.exp(-0.5*x_disk*x_disk/(100./9.))/(10.*tf.sqrt(2*np.pi)/3.))
        fy = (u * vx + v * vy + w * vz + py) - (fy_xx + fy_yy + fy_zz)
        fz = (u * wx + v * wy + w * wz + pz) - (fz_xx + fz_yy + fz_zz)

        k_xx = outer_tape.gradient(k_x, x_disk)
        k_yy = outer_tape.gradient(k_y, y_disk)
        k_zz = outer_tape.gradient(k_z, z_disk)
        eps_xx = outer_tape.gradient(eps_x, x_disk)
        eps_yy = outer_tape.gradient(eps_y, y_disk)
        eps_zz = outer_tape.gradient(eps_z, z_disk)


        # k-epsilon equations
        k_eqn = u * kx + v * ky + w * kz - (k_xx + k_yy + k_zz) - P_k + eps
        eps_eqn = (u * epsx + v * epsy + w * epsz - (eps_xx + eps_yy + eps_zz) 
                    - (C_eps_1 * P_k - C_eps_2 * eps) * (eps / (k + k_zero_correction)) - (C_eps_4 * P_k*P_k/(k + k_zero_correction)))

        return fx, fy, fz, div_u, k_eqn, eps_eqn

    @tf.function
    def train_step(self, xb_inlet, yb_inlet, zb_inlet, ub_inlet, vb_inlet, wb_inlet, kb_inlet, epsb_inlet,
                    xb_outlet, yb_outlet, zb_outlet, pb_outlet,
                    xb_bottom, yb_bottom, zb_bottom, ub_bottom, vb_bottom, wb_bottom, kb_bottom, epsb_bottom,
                    xb_top, yb_top, zb_top, ub_top, vb_top, wb_top, kb_top, epsb_top,
                    xb_front, yb_front, zb_front, ub_front, vb_front, wb_front, kb_front, epsb_front,
                    xb_back, yb_back, zb_back, ub_back, vb_back, wb_back, kb_back, epsb_back,
                    xin, yin, zin, x_disk, y_disk, z_disk):

        with tf.GradientTape(persistent=True) as tape:
            u_pred, v_pred, w_pred, _, k_pred, eps_pred = self.nn_model([tf.concat((xb_inlet,xb_bottom),axis=0),
                                                    tf.concat((yb_inlet,yb_bottom),axis=0),
                                                    tf.concat((zb_inlet,zb_bottom),axis=0)], training=True)
            
            with tf.GradientTape(persistent=True) as innertape:
                innertape.watch([yb_front, yb_back, xb_outlet, zb_top])
                u_outlet, v_outlet, w_outlet, p_outlet, k_outlet, eps_outlet = self.nn_model([xb_outlet, yb_outlet, zb_outlet], training=True)
                u_top, v_top, w_top, p_top, k_top, eps_top = self.nn_model([xb_top, yb_top, zb_top], training=True)
                u_front, v_front, w_front, p_front, k_front, eps_front = self.nn_model([xb_front, yb_front, zb_front], training=True)
                u_back, v_back, w_back, p_back, k_back, eps_back = self.nn_model([xb_back, yb_back, zb_back], training=True)

            u_loss = self.loss_fn(tf.concat((ub_inlet,ub_bottom),axis=0), u_pred)+ self.loss_fn(ub_top, u_top) 
            v_loss = self.loss_fn(tf.concat((vb_inlet,vb_bottom),axis=0), v_pred)+ self.loss_fn(vb_top, v_top) + self.loss_fn(vb_front, v_front) + self.loss_fn(vb_back, v_back) 
            w_loss = self.loss_fn(tf.concat((wb_inlet,wb_bottom),axis=0), w_pred)+ self.loss_fn(wb_top, w_top) 

            #outlet boundary
            dux_outlet = tf.reduce_mean(tf.square(innertape.gradient(u_outlet, xb_outlet)))
            dvx_outlet = tf.reduce_mean(tf.square(innertape.gradient(v_outlet, xb_outlet)))
            dwx_outlet = tf.reduce_mean(tf.square(innertape.gradient(w_outlet, xb_outlet)))
            dkx_outlet = tf.reduce_mean(tf.square(innertape.gradient(k_outlet, xb_outlet)))
            depsx_outlet = tf.reduce_mean(tf.square(innertape.gradient(eps_outlet, xb_outlet)))

            p_loss = tf.reduce_mean(tf.square(p_outlet))
            outlet_loss = dux_outlet + dvx_outlet + dwx_outlet + dkx_outlet + depsx_outlet

            #bottom boundary
            k_loss = self.loss_fn(tf.concat((kb_inlet,kb_bottom),axis=0), k_pred) + self.loss_fn(kb_top, k_top) 
            eps_loss = self.loss_fn(tf.concat((epsb_inlet,epsb_bottom),axis=0) , eps_pred) + self.loss_fn(epsb_top, eps_top) 

            #top boundary
            dpz_top = tf.reduce_mean(tf.square(innertape.gradient(p_top, zb_top)))

            top_loss =  dpz_top 

            #front boundary
            duy_front = tf.reduce_mean(tf.square(innertape.gradient(u_front, yb_front)))
            dwy_front = tf.reduce_mean(tf.square(innertape.gradient(w_front, yb_front)))
            dpy_front = tf.reduce_mean(tf.square(innertape.gradient(p_front, yb_front)))
            dky_front = tf.reduce_mean(tf.square(innertape.gradient(k_front, yb_front)))
            depsy_front = tf.reduce_mean(tf.square(innertape.gradient(eps_front, yb_front)))

            front_loss = dpy_front + duy_front  + dwy_front + dpy_front + dky_front + depsy_front 

            #back boundary
            duy_back = tf.reduce_mean(tf.square(innertape.gradient(u_back, yb_back)))
            dwy_back = tf.reduce_mean(tf.square(innertape.gradient(w_back, yb_back)))
            dpy_back = tf.reduce_mean(tf.square(innertape.gradient(p_back, yb_back)))
            dky_back = tf.reduce_mean(tf.square(innertape.gradient(k_back, yb_back)))
            depsy_back = tf.reduce_mean(tf.square(innertape.gradient(eps_back, yb_back)))

            back_loss = dpy_back + duy_back  + dwy_back + dpy_back + dky_back + depsy_back

            #boundary losses
            std_loss = u_loss + v_loss + w_loss + p_loss + k_loss  + (top_loss + front_loss + back_loss + outlet_loss)*100 + eps_loss 
            
            #domain pde losses
            fx, fy, fz, div_u, keq, epseq= self.pde_residual(tf.concat((xin),axis=0), 
                                                             tf.concat((yin),axis=0), tf.concat((zin),axis=0), training=True)
            fx_loss = tf.reduce_mean(tf.square(fx))
            fy_loss = tf.reduce_mean(tf.square(fy))
            fz_loss = tf.reduce_mean(tf.square(fz))
            div_loss = tf.reduce_mean(tf.square(div_u))
            keq_loss = tf.reduce_mean(tf.square(keq))
            epseq_loss = tf.reduce_mean(tf.square(epseq))

            #actuator disc pde losses
            fx_act, fy_act, fz_act, div_u_act, keq_act, epseq_act= self.act_disc_pde_residual(x_disk, y_disk, z_disk, training=True)
            fx_act_loss = tf.reduce_mean(tf.square(fx_act))
            fy_act_loss = tf.reduce_mean(tf.square(fy_act))
            fz_act_loss = tf.reduce_mean(tf.square(fz_act))
            div_act_loss = tf.reduce_mean(tf.square(div_u_act))
            keq_act_loss = tf.reduce_mean(tf.square(keq_act))
            epseq_act_loss = tf.reduce_mean(tf.square(epseq_act))

            residual_loss = 10*(fx_loss  + fy_loss + fz_loss) +  100*div_loss  + 1*(keq_loss) + (epseq_loss) 
            residual_act_loss = 10*(fx_act_loss  + fy_act_loss + fz_act_loss) +  100*div_act_loss  + 1*(keq_act_loss) + (epseq_act_loss) 

            #total loss
            loss = 10*std_loss + residual_loss + residual_act_loss

        #gradient and optimization
        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))
        grad_std = tape.gradient(std_loss, self.nn_model.trainable_weights)
        grad_res = tape.gradient(residual_loss, self.nn_model.trainable_weights)

        self.loss_tracker.update_state(loss)
        self.std_loss_tracker.update_state(std_loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.mlx_loss_tracker.update_state(fx_loss)
        self.mly_loss_tracker.update_state(fy_loss)
        self.mlz_loss_tracker.update_state(fz_loss)
        self.div_loss_tracker.update_state(div_loss)
        self.k_loss_tracker.update_state(keq_loss)
        self.eps_loss_tracker.update_state(epseq_loss)

        return {"loss": self.loss_tracker.result(), "std_loss": self.std_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result(), 'mom-x_loss': self.mlx_loss_tracker.result(),
                "mom-y_loss": self.mly_loss_tracker.result(),"mom-z_loss": self.mlz_loss_tracker.result(),
                "div_loss": self.div_loss_tracker.result(),
                "k_loss": self.k_loss_tracker.result(), "eps_loss": self.eps_loss_tracker.result()}, grads, \
               grad_std, grad_res

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.std_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.mlx_loss_tracker.reset_state()
        self.mly_loss_tracker.reset_state()
        self.mlz_loss_tracker.reset_state()
        self.div_loss_tracker.reset_state()
        self.Pe_loss_tracker.reset_state()
        self.k_loss_tracker.reset_state()
        self.eps_loss_tracker.reset_state()

    def run(self, epochs, proj_name, log_dir,wb=False, verbose_freq=1000, plot_freq=10000, grad_freq=5000):

        self.reset_metrics()
        history = {"loss": [], "std_loss": [], "residual_loss": [], "mom-x_loss": [], "mom-y_loss": [],"mom-z_loss": [], "div_loss": [], "k_loss": [], "eps_loss": []}

        start_time = time.time()

        for epoch in range(epochs):

            for j, ((xb_inlet, yb_inlet, zb_inlet, u_inlet, v_inlet, w_inlet, k_inlet, eps_inlet),
                    (xb_outlet, yb_outlet, zb_outlet, p_outlet),
                    (xb_bottom, yb_bottom, zb_bottom, u_bottom, v_bottom, w_bottom, k_bottom, eps_bottom),
                    (xb_top, yb_top, zb_top, u_top, v_top, w_top, k_top, eps_top),
                    (xb_front, yb_front, zb_front, u_front, v_front, w_front, k_front, eps_front),
                    (xb_back, yb_back, zb_back, u_back, v_back, w_back, k_back, eps_back),
                    (xin, yin, zin),(x_disk, y_disk, z_disk)) in enumerate(zip(self.inlet_data, self.outlet_data,
                                                 self.bottom_data, self.top_data,
                                                  self.front_data, self. back_data,
                                                    self.inner_data, self.disk_data)):
                logs, grads, grad_std, grad_res = self.train_step(xb_inlet, yb_inlet, zb_inlet, u_inlet, v_inlet, w_inlet,k_inlet, eps_inlet,
                    xb_outlet, yb_outlet, zb_outlet, p_outlet,
                    xb_bottom, yb_bottom, zb_bottom, u_bottom, v_bottom, w_bottom, k_bottom, eps_bottom,
                    xb_top, yb_top, zb_top, u_top, v_top, w_top, k_top, eps_top,
                    xb_front, yb_front, zb_front, u_front, v_front, w_front, k_front, eps_front,
                    xb_back, yb_back, zb_back, u_back, v_back, w_back, k_back, eps_back,
                    xin, yin, zin,x_disk, y_disk, z_disk)

            if (epoch+1) % grad_freq == 0:
                default_grad = tf.zeros([1])
                grad_std = [grad if grad is not None else default_grad for grad in grad_std]
                grad_res = [grad if grad is not None else default_grad for grad in grad_res]
                grads = [grad if grad is not None else default_grad for grad in grads]

            tae = time.time() - start_time
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")

        return history