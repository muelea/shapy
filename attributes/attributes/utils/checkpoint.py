import sys
import time

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class TimeoutCheckpointer(ModelCheckpoint):
    def __init__(self, exit_code: int = 3, max_duration: float = 3600, *args, **kwargs):
        super(TimeoutCheckpointer, self).__init__(*args, **kwargs)

        # Maximum possible duration
        self.max_duration = max_duration

        # The exit code to use
        self.exit_code = exit_code

        # Store the time when the object was created
        self.start_time = time.perf_counter()

    def save_checkpoint(self, *args, **kwargs):
        super(TimeoutCheckpointer, self).save_checkpoint(*args, **kwargs)
        elapsed = time.perf_counter() - self.start_time
        if self.max_duration > 0 and elapsed > self.max_duration:
            logger.warning(f'Exiting after {elapsed:.2f} seconds')
            sys.exit(self.exit_code)


def get_checkpoint_filename(network_type):

    if network_type == 'a2b':
        out_ckpt_name = 'a2b-best_model-epoch={epoch:06d}-validation_error={Loss/val:.2f}-v2v_error={Validation/v2v:.2f}-hcwh={Validation/height:.2f}-{Validation/chest:.2f}-{Validation/waist:.2f}-{Validation/hips:.2f}'
    elif network_type == 'b2a':
        out_ckpt_name = 'b2a-best_model-epoch={epoch:06d}-validation_loss={Loss/val:.2f}-average_correct_class={Loss/correct_class:.2f}'
    else:
        raise ValueError(f'Unkown type {network_type}, expected [a2b, b2a].')

    return out_ckpt_name
