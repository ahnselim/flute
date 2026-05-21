import os
import torch
import click
from typing import Callable, cast

from . import _C
from . import ops

__version__ = "0.4.2"


qgemm = cast(
    Callable[
        [
            torch.Tensor,  # inputs
            torch.Tensor,  # weight
            torch.Tensor,  # scales
            torch.Tensor,  # tables
            torch.Tensor,  # tables2
            torch.Tensor,  # workspace
            int,           # num_bits
            int,           # group_size
            int,           # template_id
            int,           # num_sms
        ],
        torch.Tensor,
    ],
    torch.ops.flute.qgemm_raw_simple,
)


qgemm_hadamard = cast(
    Callable[
        [
            torch.Tensor,  # inputs
            torch.Tensor,  # weight
            torch.Tensor,  # scales
            torch.Tensor,  # tables
            torch.Tensor,  # tables2
            torch.Tensor,  # workspace
            int,           # num_bits
            int,           # group_size
            int,           # hadamard_size
            int,           # template_id
            int,           # num_sms
        ],
        torch.Tensor,
    ],
    torch.ops.flute.qgemm_raw_simple_hadamard,
)


# Load the template configs
if os.environ.get("FLUTE_ABLATIONS", "0") == "1":
    click.secho(f"[FLUTE]: Abalations enabled", fg="yellow")
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.ablations.pth")
else:
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.pth")

if os.path.exists(TEMPLATE_CONFIGS_PATH):
    TEMPLATE_CONFIGS = torch.load(TEMPLATE_CONFIGS_PATH, weights_only=True)
    # Experimental 1-bit path: reuse the 2-bit template catalog until
    # dedicated 1-bit tuning results exist.
    if TEMPLATE_CONFIGS is not None:
        # The generated 1-bit kernel currently exposes template_id 0..35 only.
        # Reusing every 2-bit template id causes tuning to pick ids that the
        # runtime dispatch rejects with "Unsupported template_id value".
        supported_one_bit_template_ids = set(range(36))
        one_bit_configs = {}
        for (num_bits, template_id), config in TEMPLATE_CONFIGS.items():
            if (
                int(num_bits) == 2
                and int(template_id) in supported_one_bit_template_ids
                and (1, int(template_id)) not in TEMPLATE_CONFIGS
            ):
                one_bit_configs[(1, int(template_id))] = dict(config)
        TEMPLATE_CONFIGS.update(one_bit_configs)
    click.secho(f"[FLUTE]: Template configs loaded from {TEMPLATE_CONFIGS_PATH}", fg="green")
else:
    TEMPLATE_CONFIGS = None
    click.secho(f"[FLUTE]: Template configs not found at {TEMPLATE_CONFIGS_PATH}", fg="red")
