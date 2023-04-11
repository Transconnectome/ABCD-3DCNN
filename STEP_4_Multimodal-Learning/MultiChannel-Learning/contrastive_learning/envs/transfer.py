import torch

## Main function that freezes layers & re-initialize FC layer
def setting_transfer(args, net, num_unfrozen_layers): #230316 set batch norm layer requires_grad=True while finetuning!
    # freeze all of the model (for training FC layers only)
    # or specified number of layers (for finetuning only rear parts of the model)
    if num_unfrozen_layers.isnumeric(): # unfrozen_layers == 'feature_extractors':
        num_unfrozen_layers = int(num_unfrozen_layers)
    elif num_unfrozen_layers.lower() == 'all':
        print("*** Didn't freeze anything (finetune all) ***")
        return None
    else:
        raise ValueError("args.unfrozen_layers should be specified by the numbers of unfrozen layers")
    
    if num_unfrozen_layers == 0: # train FC only
        frozen_layers = 0 
        freeze_layers(net, frozen_layers)
        if (args.init_unfrozen != ''):
            initialize_weights(net, unfrozen_layers)
            print("*** initializing unfrozen layers' weights & biases are done ***")
    else: # train certain part of the model.
        for i in range(net.len_feature_extractor):
            frozen_layers, unfrozen_layers = get_freeze_criteria(net.feature_extractors[i], num_unfrozen_layers)
            freeze_layers(net, frozen_layers)
            if (args.init_unfrozen != ''):
                initialize_weights(net, unfrozen_layers)
                print("*** initializing unfrozen layers' weights & biases are done ***")
    print("*** Freezing layers for [transfer learning / finetuning] are done ***")
        

def get_freeze_criteria(feature_extractor, num_unfrozen_layers):
    layers_total = []
    for name, module in feature_extractor.named_modules():
        if is_depth1_layer(name):
            layers_total.append((name, module))
    num_layers = len(layers_total)
    freeze_until = num_layers - num_unfrozen_layers
    frozen_layers = layers_total[:freeze_until]
    unfrozen_layers = layers_total[freeze_until:]
    frozen_names = list(map(lambda x:x[0], frozen_layers))
    unfrozen_names = list(map(lambda x:x[0], unfrozen_layers))
    print(f"Frozen layers={frozen_names}\tUnfrozen layers={unfrozen_names}")

    return frozen_layers, unfrozen_layers
        
        
def is_depth1_layer(module_name):
    if module_name.count(".") == 0 and module_name:
        return True
    else:
        return False
    
# freeze non-FC layers from last layer to first layer 
def freeze_layers(net, frozen_layers, train_bn=True): 
    if frozen_layers == 0:
        frozen_layers = [('feature_extractor', net.feature_extractors)]
        
    for name, module in frozen_layers:
        module.apply(freeze)
            
    if train_bn:
        net.apply(unfreeze_bn)

    
def unfreeze_bn(layer):
    if isinstance(layer, torch.nn.BatchNorm3d):
        for params in layer.parameters():
            params.requires_grad = True
            
            
# freezing layer's parameters by setting requires_grad as False
def freeze(layer):
    for params in layer.parameters():
        params.requires_grad = False
    

# initialize FC layer 
def initialize_weights(net, unfrozen_layers):
    if unfrozen_layers == 0:
        ## initialize FC layer
        net.FClayers.apply(weight_init_kaiming_normal)
        return
    
    ## initialize other unfrozen layers
    for name, module in unfrozen_layers:
        for name, module in unfrozen_layers:
            module.apply(weight_init_kaiming_normal)

            
# initialize layer's weights & biases ref: https://jh-bk.tistory.com/10            
def weight_init_xavier_uniform(layer):
    if isinstance(layer, torch.nn.Conv3d) or isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias != None:
            layer.bias.data.fill_(0.01)
    elif isinstance(layer, torch.nn.BatchNorm3d):
        layer.weight.data.fill_(1.0)
        if layer.bias != None:
            layer.bias.data.zero_()
            
def weight_init_kaiming_normal(layer):
    if isinstance(layer, torch.nn.Conv3d) or isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias != None:
            layer.bias.data.fill_(0.01)
    elif isinstance(layer, torch.nn.BatchNorm3d):
        layer.weight.data.fill_(1.0)
        if layer.bias != None:
            layer.bias.data.zero_()