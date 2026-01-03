"""
HF Trainer callback for creating PyTorch profiling snapshots
"""

import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class PytorchProfilerCallback(TrainerCallback):
    """
    PyTorch Profiler callback.
    """

    def __init__(self, cfg, profiler_steps_wait: int = 0, profiler_steps_warmup: int = 0, profiler_steps_active: int = 5):
        # pytorch profiler steps are 1 indexed
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=profiler_steps_wait, warmup=profiler_steps_warmup, active=profiler_steps_active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(cfg.output_dir, "pytorch_profile")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        )
        self.prof.start()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.prof.step()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.prof.stop()
