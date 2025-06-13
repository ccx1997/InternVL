import torch


def load_vggt_params_in_internvl(model, vggt_path):
    vggt_state_dict = torch.load(vggt_path, map_location='cpu')
    # get target params
    vision2_model_state_dict = model.vision_model2.state_dict()
    vggt_aggregator_state_dict = {}
    for k, v in vggt_state_dict.items():
        if k.startswith('aggregator.'):
            new_k = k.replace('aggregator.', '')
            if new_k in vision2_model_state_dict:
                if v.shape == vision2_model_state_dict[new_k].shape:
                    vggt_aggregator_state_dict[new_k] = v.to(vision2_model_state_dict[new_k].dtype)
                else:
                    print(f'Shape mismatch for {new_k}: '
                          f'expect {vision2_model_state_dict[new_k].shape}, got {v.shape}')
    msg = model.vision_model2.load_state_dict(vggt_aggregator_state_dict, strict=True)
    print(f'Loaded VGGT aggregator parameters into vision_model2: {msg}')



