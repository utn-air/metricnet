import warnings

warnings.simplefilter("ignore")

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from metricnet.benchmark.models_utils import get_metricNet, load_config, load_model


class NavAgent:
    def __init__(self, model_cfg_path=None, ckpt_path=None, device="cpu"):
        assert model_cfg_path is not None, "Model config is None"
        assert ckpt_path is not None, "Checkpoint is None"
        self.model_cfg = load_config(model_cfg_path)
        self.model = load_model(ckpt_path, self.model_cfg, device=device)
        self.model.to(device)
        self.model.eval()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_cfg["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        if self.model_cfg.get("scale", {}).get("enabled", False):
            self.metricNet = get_metricNet(self.model_cfg, device)
            self.metricNet.eval()
