import copy
import torch,random
from tqdm import tqdm
from .utils import alpha, dalpha_dt, d_1_minus_alpha_sq_dt
from scipy import integrate
import functools
import numpy as np

class BaseFlow():
	def __init__(self, device, model=None, ema_model=None, num_steps=1000):
		self.model = model
		self.ema_model = ema_model
		self.N = num_steps
		self.device = device


	@torch.no_grad()
	def sample_ode(self, z0=None, N=None):
		### NOTE: Use Euler method to sample from the learned flow
		if N is None:
			N = self.N    
		dt = 1./N
		traj = [] # to store the trajectory
		z = z0.detach().clone()
		batchsize = z.shape[0]
		
		traj.append(z.detach().clone())
		for i in range(N):
			t = torch.ones((batchsize,1), device=self.device) * i / N
			pred = self.model(t.squeeze(), z)
			z = z.detach().clone() + pred * dt
			traj.append(z.detach().clone())
		return traj

	@torch.no_grad()
	def sample_ode_generative(self, z1=None, N=None, use_tqdm=True, solver = 'euler', model_kwargs={}):
		assert solver in ['euler', 'heun']
		tq = tqdm if use_tqdm else lambda x: x
		if N is None:
			N = self.N    
		if solver == 'heun' and N % 2 == 0:
			raise ValueError("N must be odd when using Heun's method.")
		if solver == 'heun':
			N = (N + 1) // 2
		dt = -1./N
    
		traj = [] # to store the trajectory
		x0hat_list = []
		z = z1.detach().clone()
		batchsize = z.shape[0]
		traj.append(z.detach().clone())
    
		for i in tq(reversed(range(1,N+1))):
			t = torch.ones((batchsize,1), device=self.device) * i / N
			t_next = torch.ones((batchsize,1), device=self.device) * (i-1) / N
			vt = self.model(t.squeeze(), z, **model_kwargs)
			if solver == 'heun' and i > 1:
				z_next = z.detach().clone() + vt * dt
				vt_next = self.model(t_next.squeeze(), z_next, **model_kwargs)
				vt = (vt + vt_next) / 2
			x0hat = z - vt * t.view(-1,1,1,1)
			x0hat_list.append(x0hat)
			z = z.detach().clone() + vt * dt
			traj.append(z.detach().clone())

		return traj, x0hat_list

  
	def sample_ode_generative_bbox(self, z1=None, N=None, use_tqdm=True, solver = 'RK45', eps = 1e-3, rtol=1e-5, atol=1e-5,if_pred_x0=False):
		dshape = z1.shape
		device = z1.device
		def ode_func(t, x):
			x = torch.from_numpy(x.reshape(dshape)).to(device).type(torch.float32)
			vec_t = torch.ones(dshape[0], device=x.device) * t
			if if_pred_x0:
				vt = z1 - self.model(vec_t, x)
			else:
				vt = self.model(vec_t, x)
			vt = vt.detach().cpu().numpy().reshape(-1)
			return vt
		solution = integrate.solve_ivp(ode_func, (1, eps), z1.detach().cpu().numpy().reshape(-1), method=solver, rtol = rtol, atol = atol)
		nfe = solution.nfev
		result = torch.from_numpy(solution.y[:,-1].reshape(dshape))
		return result, nfe


class RectifiedFlow(BaseFlow):
    def __init__(self, discrete=False, **kwargs):
        super().__init__(**kwargs)
        self.discrete = discrete

    def get_train_tuple_flow(self, z0=None, z1=None, t = None, eps=1e-5):
        if t is None:
            if self.discrete:
                t = torch.randint(1,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
            else:
                t = torch.rand((z1.shape[0], 1), device=self.device)
                t = t * (1 - eps) + eps

        if len(z1.shape) == 2:
            z_t =  t * z1 + (1.-t) * z0
        elif len(z1.shape) == 4:
            t = t.view(-1, 1, 1, 1)
            z_t =  t * z1 + (1.-t) * z0
            t = t.view(-1)
        else:
            raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
        target = z1 - z0
        return z_t, t, target

    def get_train_tuple_ddpm(self, z0=None, z1=None, t = None):
        a = 19.9
        b = 0.1
        if t is None:
            t = torch.rand((z1.shape[0], 1), device=self.device)
            eps = 1e-5
            t = t * (1 - eps) + eps
        alpha_t = alpha(t)
        if len(z1.shape) == 2:
            z_t =  torch.sqrt(1 - alpha_t ** 2) * z1 + alpha_t * z0
        elif len(z1.shape) == 4:
            raise NotImplementedError
            t = t.view(-1, 1, 1, 1)
            z_t =  t * z1 + (1.-t) * z0
        else:
            raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
        target = d_1_minus_alpha_sq_dt(t) * z1 + dalpha_dt(t) * z0
        return z_t, t, target


class ConsistencyFlow(RectifiedFlow):
    def __init__(self, device, model, ema_model, threshold, trunc_threshold, pretrained_model=None):
        self.ema_model = ema_model
        self.pretrained_model = pretrained_model # copy.deepcopy(model.module)
        self.model = model
        self.device = device
        self.threshold = threshold
        self.trunc_threshold = trunc_threshold

    def get_train_tuple(self, z0=None, z1=None, t=None, eps=1e-5, model_kwargs={}):                     
        t = t.view(-1, 1, 1, 1)
        zt = t * z1 + (1. - t) * z0
        t = t.squeeze()
        gt_flow = z1 - z0
        # local consistency
        t_down = copy.deepcopy(t)
        t_up = 1 - copy.deepcopy(t)
        t_up[t_up >= self.threshold] = self.threshold
        t_down[t_down >= self.threshold] = self.threshold
        delta_t_down = (torch.rand_like(t)*t_down).to(t.device)
        delta_t_up = (torch.rand_like(t)*t_up).to(t.device)
        post_t = t - delta_t_down
        prev_t = t + delta_t_up
        if self.pretrained_model is not None:
            with torch.no_grad():
                teacher_vt = self.pretrained_model(t, zt, y=model_kwargs.get("y", None))
                post_zt = zt - delta_t_down.view(-1, 1, 1, 1) * teacher_vt
                prev_zt = zt + delta_t_up.view(-1, 1, 1, 1) * teacher_vt
        else:
            post_zt = zt - delta_t_down.view(-1, 1, 1, 1) * (z1 - z0)
            prev_zt = zt + delta_t_up.view(-1, 1, 1, 1) * (z1 - z0)     
        
        curr_vt = self.model(t, zt, **model_kwargs)
        curr_z0 = zt - t.view(-1, 1, 1, 1) * curr_vt
        with torch.no_grad():
            # consistency post_t            
            post_vt = self.ema_model(post_t, post_zt, **model_kwargs).detach()
            prev_vt = self.ema_model(prev_t, prev_zt, **model_kwargs).detach()
            post_z0 = post_zt - post_t.view(-1, 1, 1, 1) * post_vt
            prev_z0 = prev_zt - prev_t.view(-1, 1, 1, 1) * prev_vt
            # reflow: predict z0 directly from z1
        reflow_v1 = self.model(torch.ones_like(t), z1)
        reflow_z0 = z1 - reflow_v1
        # compute reflow intermidiate state, should we replace with ema
        # here there are few ideas, ema for reflow_v1/vt (ema for no grad, call ema version) and optimize one of them (call swap version) -- not work)
        reflow_zt =  t.view(-1, 1, 1, 1) * reflow_z0.detach() + (1 - t).view(-1, 1, 1, 1) * z1
        reflow_vt = self.model(t, reflow_zt, **model_kwargs)
        reflow_z0_rescon = reflow_zt - t.view(-1, 1, 1, 1)*reflow_vt
        # song technique to truncate
        post_z0[post_t<self.trunc_threshold] = z0[post_t<self.trunc_threshold]
        return curr_vt, post_vt, prev_vt, gt_flow, curr_z0, post_z0, prev_z0, reflow_z0_rescon, reflow_z0
