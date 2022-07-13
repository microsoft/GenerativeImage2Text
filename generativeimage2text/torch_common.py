from pprint import pformat
import logging
import torch
import io
from azfuse import File


def recursive_to_device(d, device, **kwargs):
    if isinstance(d, tuple) or isinstance(d, list):
        return [recursive_to_device(x, device, **kwargs) for x in d]
    elif isinstance(d, dict):
        return dict((k, recursive_to_device(v, device)) for k, v in d.items())
    elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):
        #return d.to(device, non_blocking=True)
        return d.to(device, **kwargs)
    else:
        return d

def resize_2d_pos_embed(origin_pos_embed, origin_input, patch_size, after_input):
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = origin_input // patch_size
    assert (origin_input % patch_size) == 0
    grid_after = after_input // patch_size
    assert (after_input % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]
    assert origin_pos_embed.shape[1] == grid_before * grid_before + 1

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size, mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    return pos_embed

def torch_load(filename):
    with File.open(filename, 'rb') as fp:
        buf = io.BytesIO(fp.read())
    result = torch.load(buf, map_location=lambda storage, loc: storage)
    return result

def remove_prefix(model, prefix):
    out = {}
    for k, v in model.items():
        while k.startswith(prefix):
            k = k[len(prefix): ]
        out[k] = v
    return out

def strip_prefix_if_present(state_dict, prefix):
    return remove_prefix(state_dict, prefix)

def load_model_state_ignore_mismatch(model, init_dict):
    real_init_dict = {}
    name_to_param = dict(model.named_parameters())
    name_to_param.update(dict(model.named_buffers()))

    def same_shape(a, b):
        return len(a.shape) == len(b.shape) and \
                all(x == y for x, y in zip(a.shape, b.shape))

    num_ignored = 0
    unique_key_in_init_dict = []
    keys_shape_mismatch = []
    for k in init_dict:
        if k in name_to_param:
            if same_shape(init_dict[k], name_to_param[k]):
                real_init_dict[k] = init_dict[k]
            else:
                logging.info('{} shape is not consistent, expected: {}; got '
                             '{}'.format(k, name_to_param[k].shape, init_dict[k].shape))
                keys_shape_mismatch.append(k)
        else:
            unique_key_in_init_dict.append(k)
            num_ignored = num_ignored + 1

    logging.info('unique keys in init dict = {}; total = {}'.format(
        pformat(unique_key_in_init_dict), len(unique_key_in_init_dict),
    ))
    result = model.load_state_dict(real_init_dict, strict=False)
    logging.info('unique key (not initialized) in current model = {}'.format(
        pformat(result.missing_keys),
    ))

    #logging.info('loaded key = {}'.format(
        #pformat(list(real_init_dict.keys()))))

def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    load_model_state_ignore_mismatch(model, model_state_dict)

def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} will be loaded from {: <{}} of shape {}"
    target_source_name_matched = 0
    all_key_old = set()
    updated_keys = []
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        updated_keys.append(key)
        all_key_old.add(key_old)
        target_source_name_matched += 1
        logging.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )
    logging.info('target model param = {}; name matched = {}; loaded = {}'.format(
        len(model_state_dict), target_source_name_matched,
        len(loaded_state_dict)))
    logging.info('from loaded; ignore = {}'.format(
        pformat([k for k in loaded_state_dict if k not in all_key_old])))
    updated_keys = set(updated_keys)
    no_update_keys = [k for k in model_state_dict.keys() if k not in updated_keys]
    for k in no_update_keys:
        del model_state_dict[k]

