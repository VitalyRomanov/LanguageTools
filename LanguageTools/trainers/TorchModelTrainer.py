from abc import ABC

from LanguageTools.trainers.AbstractModelTrainer import AbstractModelTrainer


try:
    import torch
except ImportError:
    raise ImportError("Install Torch: pip install torch")


class TorchModelTrainer(AbstractModelTrainer, ABC):
    def set_gpu(self):
        if self.gpu_id != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.use_cuda = True
            self.device = f"cuda:{self.gpu_id}"
        else:
            self.use_cuda = False
            self.device = "cpu"

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(self, model, path):
        model.load_state_dict(torch.load(self.ckpt_path.joinpath("checkpoint"), map_location=torch.device('cpu')))

    def _create_summary_writer(self, path):
        from torch.utils.tensorboard import SummaryWriter
        self.summary_writer = SummaryWriter(path)

    def _write_to_summary(self, label, value, step):
        self.summary_writer.add_scalar(label, value, step)

    def _create_optimizer(self, model):
        parameters = set(model.parameters())
        self.optimizer = torch.optim.AdamW(
            list(parameters),  # model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0 if self.weight_decay is None else self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)

    def _lr_scheduler_step(self):
        self.scheduler.step()

    @staticmethod
    def set_model_training(model):
        model.train()

    @staticmethod
    def set_model_evaluation(model):
        model.eval()