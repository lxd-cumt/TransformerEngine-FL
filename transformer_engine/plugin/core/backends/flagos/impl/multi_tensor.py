# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
from torch.distributed._tensor import DTensor
import flag_gems
from contextlib import nullcontext



def multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    try:
        flag_gems_global_registrar = getattr(flag_gems, 'current_work_registrar', None)
    except Exception as e:
        raise RuntimeError(f"Failed to get flag gems registrar: {e}.")
    is_flag_gems_global_enabled = flag_gems_global_registrar is not None
    # Use nullcontext if flag_gems is already globally enabled, otherwise use use_gems() context
    gems_context = nullcontext() if is_flag_gems_global_enabled else flag_gems.use_gems()

    with gems_context:
        tensors = tensor_lists[0]

        if per_tensor:
            norms = [torch.norm(t.float(), p=2) for t in tensors]
            return norms, None
        else:
            total_norm_sq = sum(torch.sum(t.float() ** 2) for t in tensors)
            total_norm = torch.sqrt(total_norm_sq)
            return total_norm, None


def multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale):
    try:
        flag_gems_global_registrar = getattr(flag_gems, 'current_work_registrar', None)
    except Exception as e:
        raise RuntimeError(f"Failed to get flag gems registrar: {e}.")
    is_flag_gems_global_enabled = flag_gems_global_registrar is not None
    # Use nullcontext if flag_gems is already globally enabled, otherwise use use_gems() context
    gems_context = nullcontext() if is_flag_gems_global_enabled else flag_gems.use_gems()

    with gems_context:
        for src, dst in zip(tensor_lists[0], tensor_lists[1]):
            dst.copy_(src * scale)
