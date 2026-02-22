# Checkpoint loading for task expansion (see doc below)
import torch
from collections import OrderedDict

DEFAULT_TASK_PARAM_PATTERNS = ["task_embed", "task_emb", "prompt_embed", "prompt_emb", "task_prompt"]


def _is_task_param(key, shape, old_num_tasks, patterns):
    if len(shape) == 0 or shape[0] != old_num_tasks:
        return False
    return any(p.lower() in key.lower() for p in patterns)


def load_pretrained_ckpt_with_task_expansion(
    model, ckpt_path, old_num_tasks=5, new_num_tasks=8, copy_from_task_index=0,
    task_param_patterns=None, map_location=None, strict=False,
):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
    if task_param_patterns is None:
        task_param_patterns = DEFAULT_TASK_PARAM_PATTERNS
    model_state = model.state_dict()
    loaded, missing, unexpected, expanded = [], list(model_state.keys()), [], []

    for key, ckpt_tensor in state_dict.items():
        mkey = key[4:] if key.startswith("net.") and key[4:] in model_state else key
        if mkey not in model_state:
            unexpected.append(key)
            continue
        model_tensor = model_state[mkey]
        cs, ms = list(ckpt_tensor.shape), list(model_tensor.shape)
        if cs == ms:
            model_state[mkey] = ckpt_tensor.clone()
            loaded.append(mkey)
            if mkey in missing:
                missing.remove(mkey)
            continue
        if _is_task_param(mkey, cs, old_num_tasks, task_param_patterns) and len(ms) >= 1 and ms[0] == new_num_tasks:
            copy_idx = min(max(0, copy_from_task_index), old_num_tasks - 1) if old_num_tasks > 0 else 0
            new_tensor = model_tensor.clone()
            new_tensor[:old_num_tasks] = ckpt_tensor
            for i in range(old_num_tasks, new_num_tasks):
                new_tensor[i] = ckpt_tensor[copy_idx].clone()
            model_state[mkey] = new_tensor
            loaded.append(mkey)
            expanded.append(mkey)
            if mkey in missing:
                missing.remove(mkey)
        else:
            unexpected.append(key)

    model.load_state_dict(model_state, strict=False)
    return {"loaded": loaded, "missing": missing, "unexpected": unexpected, "expanded": expanded}


def load_adair_ckpt_for_uie_finetune(model_or_lightning_module, ckpt_path, old_num_tasks=5, new_num_tasks=8, copy_from_task_index=0, map_location=None):
    target = getattr(model_or_lightning_module, "net", model_or_lightning_module)
    return load_pretrained_ckpt_with_task_expansion(
        target, ckpt_path, old_num_tasks=old_num_tasks, new_num_tasks=new_num_tasks,
        copy_from_task_index=copy_from_task_index, map_location=map_location, strict=False,
    )
