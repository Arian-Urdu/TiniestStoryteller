try:
    from config import freeze_up_to
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import freeze_up_to

def freeze_layers(model):
    for name, param in model.named_parameters():
        if "blocks" not in name:
            continue
        layer_num = int(name.split('.')[1])
        if layer_num <= freeze_up_to:
            param.requires_grad = False

def unfreeze_layer(model, layer_id):
    for name, param in model.named_parameters():
        if "blocks" not in name:
            continue
        layer_num = int(name.split('.')[1])
        if layer_num == layer_id:
            param.requires_grad = True