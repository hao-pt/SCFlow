import copy

import torch
from scipy import integrate
from tqdm import tqdm

from sd_distill.model_utils import alpha, d_1_minus_alpha_sq_dt, dalpha_dt


class BaseFlow:
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
        dt = 1.0 / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1), device=self.device) * i / N
            pred = self.model(z, t.squeeze())
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj

    @torch.no_grad()
    def sample_ode_generative(
        self, z1=None, N=None, use_tqdm=True, solver="euler", **model_kwargs,
    ):
        assert solver in ["euler", "heun"]
        tq = tqdm if use_tqdm else lambda x: x
        if N is None:
            N = self.N
        if solver == "heun" and N % 2 == 0:
            raise ValueError("N must be odd when using Heun's method.")
        if solver == "heun":
            N = (N + 1) // 2
        dt = -1.0 / N

        traj = []  # to store the trajectory
        x0hat_list = []
        z = z1.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())

        for i in tq(reversed(range(1, N + 1))):
            t = torch.ones((batchsize, 1), device=self.device, dtype=z1.dtype) * i / N
            t_next = (
                torch.ones((batchsize, 1), device=self.device, dtype=z1.dtype)
                * (i - 1)
                / N
            )
            vt = self.model(z, t.squeeze(), **model_kwargs).sample
            if solver == "heun" and i > 1:
                z_next = z.detach().clone() + vt * dt
                vt_next = self.model(z_next, t_next.squeeze(), **model_kwargs).sample
                vt = (vt + vt_next) / 2
            x0hat = z - vt * t.view(-1, 1, 1, 1)
            x0hat_list.append(x0hat)
            z = z.detach().clone() + vt * dt
            traj.append(z.detach().clone())

        return traj, x0hat_list

    def sample_ode_generative_bbox(
        self,
        z1=None,
        N=None,
        use_tqdm=True,
        solver="RK45",
        eps=1e-3,
        rtol=1e-5,
        atol=1e-5,
        if_pred_x0=False,
    ):
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

        solution = integrate.solve_ivp(
            ode_func,
            (1, eps),
            z1.detach().cpu().numpy().reshape(-1),
            method=solver,
            rtol=rtol,
            atol=atol,
        )
        nfe = solution.nfev
        result = torch.from_numpy(solution.y[:, -1].reshape(dshape))
        return result, nfe


class RectifiedFlow(BaseFlow):
    def __init__(self, discrete=False, **kwargs):
        super().__init__(**kwargs)
        self.discrete = discrete

    def get_train_tuple_flow(self, z0=None, z1=None, t=None, eps=1e-5):
        if t is None:
            if self.discrete:
                t = (
                    torch.randint(1, self.TN + 1, (z0.shape[0],)).to(z1.device).float()
                    / self.TN
                )
            else:
                t = torch.rand((z1.shape[0], 1), device=self.device)
                t = t * (1 - eps) + eps

        if len(z1.shape) == 2:
            z_t = t * z1 + (1.0 - t) * z0
        elif len(z1.shape) == 4:
            t = t.view(-1, 1, 1, 1)
            z_t = t * z1 + (1.0 - t) * z0
            t = t.view(-1)
        else:
            raise NotImplementedError(
                f"get_train_tuple not implemented for {self.__class__.__name__}.",
            )
        target = z1 - z0
        return z_t, t, target

    def get_train_tuple_ddpm(self, z0=None, z1=None, t=None):
        if t is None:
            t = torch.rand((z1.shape[0], 1), device=self.device)
            eps = 1e-5
            t = t * (1 - eps) + eps
        alpha_t = alpha(t)
        if len(z1.shape) == 2:
            z_t = torch.sqrt(1 - alpha_t**2) * z1 + alpha_t * z0
        elif len(z1.shape) == 4:
            raise NotImplementedError
            t = t.view(-1, 1, 1, 1)
            z_t = t * z1 + (1.0 - t) * z0
        else:
            raise NotImplementedError(
                f"get_train_tuple not implemented for {self.__class__.__name__}.",
            )
        target = d_1_minus_alpha_sq_dt(t) * z1 + dalpha_dt(t) * z0
        return z_t, t, target


class ConsistencyFlow(RectifiedFlow):
    def __init__(
        self,
        device,
        model,
        ema_model,
        threshold,
        trunc_threshold,
        pretrained_model=None,
        noise_scheduler=None,
        post_conditioning_outputs=False,
        student_guidance=False,
    ):
        self.ema_model = ema_model
        self.pretrained_model = pretrained_model  # copy.deepcopy(model.module)
        self.model = model
        self.device = device
        self.threshold = threshold
        self.trunc_threshold = trunc_threshold
        self.noise_scheduler = noise_scheduler
        self.post_conditioning_outputs = post_conditioning_outputs
        self.student_guidance = student_guidance

    def _noise_forward(self, latent, noise, t):
        t_scaled = t / self.noise_scheduler.config.num_train_timesteps
        return (
            t_scaled.view(-1, 1, 1, 1) * noise
            + (1.0 - t_scaled.view(-1, 1, 1, 1)) * latent
        )

    def _noise2data(self, zt, t, pred_eps):
        # ###DDPM
        # alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(zt.device, dtype=zt.dtype)
        # alpha_cumprod_t = extract(alphas_cumprod, t, zt.shape)
        # pred_z0 = (zt - (1 - alpha_cumprod_t) ** (0.5) * pred_eps) / alpha_cumprod_t ** (0.5)

        # ###RectifiedFlow
        T = self.noise_scheduler.config.num_train_timesteps
        pred_z0 = zt + t.view(-1, 1, 1, 1) / T * pred_eps
        return pred_z0

    def _ddim_posterior_sample(self, pred_z0, pred_eps, t, t_next, ddim_eta=0.0):
        # q_sigma(x_t-1|x_t, x_0)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            pred_z0.device, dtype=pred_z0.dtype,
        )
        # alphas_cumprod_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]], dim=0)
        alpha_cumprod_t = extract(alphas_cumprod, t, pred_z0.shape)
        alpha_cumprod_t_next = extract(alphas_cumprod, t_next, pred_z0.shape)
        sigma_t = ddim_eta * torch.sqrt(
            (1 - alpha_cumprod_t_next)
            / (1 - alpha_cumprod_t)
            * (1 - alpha_cumprod_t / alpha_cumprod_t_next),
        )

        # direction pointing to x_t
        dir_xt = (1.0 - alpha_cumprod_t_next - sigma_t**2).sqrt() * pred_eps
        noise = sigma_t * torch.randn_like(pred_z0)
        x_prev = alpha_cumprod_t_next.sqrt() * pred_z0 + dir_xt + noise

        return x_prev

    def _predict_cond(self, unet, zt, t, prompt_embeds):
        pred_eps = unet(
            zt, t, encoder_hidden_states=prompt_embeds,
        ).sample  # .to(dtype=zt.dtype)
        pred_z0 = self._noise2data(zt, t, pred_eps)
        if self.post_conditioning_outputs:
            pred_z0 = post_conditioning(zt, t, pred_z0)

        return pred_z0, pred_eps

    def _get_z0_with_guidance(self, unet, zt, t, guidance_scale, model_kwargs):
        uncond_eps = unet(
            zt, t, encoder_hidden_states=model_kwargs.get("prompt_null_embeds", None),
        ).sample
        cond_eps = unet(
            zt, t, encoder_hidden_states=model_kwargs.get("prompt_embeds", None),
        ).sample
        pred_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        pred_z0 = self._noise2data(zt, t, pred_eps)
        return pred_z0, pred_eps

    def _predict(self, unet, zt, t, guidance_scale, model_kwargs, use_uncond=False):
        # combined of _predict_cond and _get_z0_with_guidance functions
        cond_eps = unet(
            zt, t, encoder_hidden_states=model_kwargs.get("prompt_embeds", None),
        ).sample  # .to(dtype=zt.dtype)
        if use_uncond:
            uncond_eps = unet(
                zt,
                t,
                encoder_hidden_states=model_kwargs.get("prompt_null_embeds", None),
            ).sample  # .to(dtype=zt.dtype)
            pred_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        else:
            pred_eps = cond_eps
        pred_z0 = self._noise2data(zt, t, pred_eps)
        return pred_z0, pred_eps

    def get_train_tuple(self, z0, z1, t, eps=1e-5, guidance_scale=7.5, model_kwargs={}):
        # Notation: z0 is data, z1 is noise
        # #### DDPM
        T = self.noise_scheduler.config.num_train_timesteps
        t_scaled = t / T
        # zt = self.noise_scheduler.add_noise(z0, z1, t)
        zt = t_scaled.view(-1, 1, 1, 1) * z1 + (1.0 - t_scaled.view(-1, 1, 1, 1)) * z0
        # gt_flow = z1 - z0

        # local consistency
        t_downs = copy.deepcopy(t_scaled)
        t_ups = 1 - copy.deepcopy(t_scaled)
        t_ups[t_ups >= self.threshold] = self.threshold
        t_downs[t_downs >= self.threshold] = self.threshold

        delta_t_down = (torch.rand_like(t_scaled) * t_downs).to(t.device)
        delta_t_up = (torch.rand_like(t_scaled) * t_ups).to(t.device)
        post_t = t_scaled - delta_t_down
        prev_t = t_scaled + delta_t_up

        post_t_scaled = post_t

        # scale t back to 1000
        post_t = (post_t * T).long()
        prev_t = (prev_t * T).long()

        # # delta_t_down = torch.cat([torch.randint(0, t_down+1, (1,)) for t_down in t_downs]).to(t.device)
        # # delta_t_up = torch.cat([torch.randint(0, t_up+1, (1,)) for t_up in t_ups]).to(t.device)
        # delta_t_down = torch.randint(0, self.threshold, (t.size(0),)).to(t.device)
        # delta_t_up = torch.randint(0, self.threshold, (t.size(0),)).to(t.device)
        # post_t = torch.clamp(t - delta_t_down, 0, T-1).long()
        # prev_t = torch.clamp(t + delta_t_up, 0, T-1).long()

        with torch.no_grad():
            teacher_z0, teacher_eps = self._predict(
                self.pretrained_model,
                zt,
                t,
                guidance_scale,
                model_kwargs,
                use_uncond=True,
            )
            # torchvision.utils.save_image(teacher_z0_pred[:, :3], f"teacher_x0_t{timesteps.detach().cpu().item()}.jpg")
            # post_zt = self.noise_scheduler.add_noise(teacher_z0, teacher_eps, post_t)
            # prev_zt = self.noise_scheduler.add_noise(teacher_z0, teacher_eps, prev_t)
            post_zt = zt + delta_t_down.view(-1, 1, 1, 1) * teacher_eps
            prev_zt = zt - delta_t_up.view(-1, 1, 1, 1) * teacher_eps

        curr_z0, curr_eps = self._predict(
            self.model,
            zt,
            t,
            guidance_scale,
            model_kwargs,
            use_uncond=self.student_guidance,
        )
        # torchvision.utils.save_image(curr_z0[:, :3], f"curz0_t{t.detach().cpu().item()}.jpg")
        with torch.no_grad():
            post_z0, post_eps = self._predict(
                self.ema_model,
                post_zt,
                post_t,
                guidance_scale,
                model_kwargs,
                use_uncond=self.student_guidance,
            )
            prev_z0, prev_eps = self._predict(
                self.ema_model,
                prev_zt,
                prev_t,
                guidance_scale,
                model_kwargs,
                use_uncond=self.student_guidance,
            )

            # post_z0, post_eps = self._predict_cond(self.ema_model, post_zt, post_t, model_kwargs.get("prompt_embeds", None))
            # prev_z0, prev_eps = self._predict_cond(self.ema_model, prev_zt, prev_t, model_kwargs.get("prompt_embeds", None))

        # reflow: predict z0 directly from z1
        # ### DDPM
        reflow_z0, reflow_eps = self._predict(
            self.model,
            z1,
            torch.full_like(t, T),
            guidance_scale,
            model_kwargs,
            use_uncond=self.student_guidance,
        )
        # reflow_z0, reflow_eps = self._predict(self.model, z1, torch.ones_like(t), guidance_scale, model_kwargs, use_uncond=self.student_guidance)

        # compute reflow intermidiate state, should we replace with ema
        # here there are few ideas, ema for reflow_v1/vt (ema for no grad, call ema version) and optimize one of them (call swap version) -- not work)
        # ###DDPM
        # reflow_zt = self.noise_scheduler.add_noise(reflow_z0.detach(), reflow_eps, t)
        # reflow_z0_rescon, reflow_eps_rescon = self._predict(self.model, reflow_zt, t, guidance_scale, model_kwargs, use_uncond=self.student_guidance)
        # reflow_z0_rescon = torch.zeros_like(curr_z0)
        # reflow_z0_rescon, reflow_eps_rescon = self._predict(self.pretrained_model, reflow_zt, t, guidance_scale, model_kwargs, use_uncond=True)

        # ####RectifiedFlow
        reflow_zt = (
            t_scaled.view(-1, 1, 1, 1) * z1
            + (1 - t_scaled).view(-1, 1, 1, 1) * reflow_z0.detach()
        )
        reflow_z0_rescon, reflow_eps_rescon = self._predict(
            self.model,
            reflow_zt,
            t,
            guidance_scale,
            model_kwargs,
            use_uncond=self.student_guidance,
        )
        reflow_z0_rescon = reflow_zt + t_scaled.view(-1, 1, 1, 1) * reflow_eps_rescon

        # song technique to truncate
        post_z0[post_t_scaled < self.trunc_threshold] = z0[
            post_t_scaled < self.trunc_threshold
        ].float()
        return (
            curr_z0,
            post_z0.detach(),
            prev_z0.detach(),
            reflow_z0_rescon,
            reflow_z0,
            post_t,
        )
        # return curr_z0, post_z0.detach(), 0., reflow_eps_rescon, 0.
        # return curr_z0, post_z0.detach(), 0., 0., 0.
        # return curr_z0, post_z0.detach(), 0., 0., reflow_z0

    def ddim_sample_loop_skip(
        self,
        args,
        unet,
        noise=None,
        weight_dtype=None,
        prompt_embed=None,
        prompt_null_embed=None,
        guidance_scale=4.5,
        progress=False,
        eta=0.0,
        skip=50,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        device = next(self.model.parameters()).device
        shape = (1, 4, args.resolution // 8, args.resolution // 8)
        T = self.noise_scheduler.config.num_train_timesteps
        if skip is None:
            skip = int(T // args.num_sample_timesteps)
            num_inference_steps = args.num_sample_timesteps
        else:
            num_inference_steps = T // skip

        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device, dtype=weight_dtype)

        # indices = list(range(T-1, -1, -skip))
        # indices_next = list(indices[1:]) + [0]

        # NOTE: DDPM uses discrete step from [999, 0] so RectifiedFlow still uses discrete times for UNet.
        # However, its timerange is from 1000 to 0.
        # Rectified goes from t=0 (noise) to t=1 (data) so dt should always be positive

        indices = [
            (1.0 - i / num_inference_steps) * T for i in range(num_inference_steps)
        ]
        indices_next = indices[1:] + [0]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        z0_list = []
        sample_list = [noise]

        model_kwargs = {
            "prompt_null_embeds": prompt_null_embed,
            "prompt_embeds": prompt_embed,
        }
        use_uncond = False if prompt_null_embed is None else True
        dt = 1.0 / num_inference_steps
        for i, j in zip(indices, indices_next):
            t = torch.tensor([i] * shape[0], device=device, dtype=torch.float)
            with torch.no_grad():
                # pred_z0, pred_eps = self.get_z0_with_guidance(unet, img, t, guidance_scale, model_kwargs)
                pred_z0, pred_eps = self._predict(
                    unet, img, t, guidance_scale, model_kwargs, use_uncond=use_uncond,
                )
                # sample = self.noise_scheduler.add_noise(pred_z0, pred_eps, t_next)
                # sample = self._ddim_posterior_sample(pred_z0, pred_eps, t, t_next)
                sample = img + dt * pred_eps

                sample_list.append(sample)
                z0_list.append(pred_z0)
                img = sample

        return sample_list, z0_list


def extract(input, t, shape):
    t = t.long()
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less",
        )
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    # original CD: c_out = scaled_timestep * sigma_data / (scaled_timestep**2 + sigma_data**2) ** 0.5
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def post_conditioning(input_zt, timestep, pred_z0):
    c_skip, c_out = scalings_for_boundary_conditions(
        timestep, sigma_data=0.5, timestep_scaling=10.0,
    )
    c_skip, c_out = [
        append_dims(x, input_zt.ndim)
        for x in [c_skip.to(dtype=pred_z0.dtype), c_out.to(dtype=pred_z0.dtype)]
    ]
    model_pred = c_skip * input_zt + c_out * pred_z0
    return model_pred
