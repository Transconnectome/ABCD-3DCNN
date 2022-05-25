import torch

## Main function that freezes layers & re-initialize FC layer
def setting_transfer(net, num_unfreezed):
    if num_unfreezed != 0:
        layers_total = []

        for name, module in net.features.named_modules():
            if is_layer(name):
                layers_total.append((name, module))
                
        num_layers = len(layers_total)

        freeze_until = num_layers - num_unfreezed
        frozen_layers = layers_total[:freeze_until]
        unfrozen_layers = layers_total[freeze_until:]
        
        freeze_layers(net, frozen_layers)
        initialize_weights(net, unfrozen_layers)
        
    elif num_unfreezed == 0:
        freeze_layers(net, None)
        initialize_weights(net, None)
        
    else:
        print("ERROR!! Invalid freeze layer number")
        
        
# Check whether a module is layer or not
# ex) "Sequential" is a total Model, "relu0", "pool0" are not a layer, "denseblock1,2,3,4" are blocks(set of layers),
#     "transition1.conv", "transition1.norm", "denseblock1.denselayer1.conv0" and so on are sub-layers.
#     "conv0", "norm0" are layers. [transition1, denseblock1.denselayer1~6, denseblock2.denselayer1~12] are layers.
def is_layer(module_name):
    if ((module_name.count(".") == 1 and "layer" in module_name) or
        (('.' not in module_name) and (True in map(lambda x: x in module_name, ['conv','norm','transition'])))
       ):
        return True
    else:
        return False

    
# freeze non-FC layers from last layer to first layer 
def freeze_layers(net, frozen_layers):    
    if frozen_layers == None:
        net.features.apply(freeze)
    else:
        for name, module in frozen_layers:
            module.apply(freeze)

            
# freezing layer's parameters by setting requires_grad as False
def freeze(layer):
    for params in layer.parameters():
        params.requires_grad = False
    

# initialize FC layer 
def initialize_weights(net, unfrozen_layers):
    ## initialize FC layer
    net.FClayers.apply(weight_init_xavier_uniform)
    
    if unfrozen_layers == None:
        return
    
    ## initialize other unfrozen layers
    for name, module in unfrozen_layers:
        for name, module in unfrozen_layers:
            module.apply(weight_init_xavier_uniform)

            
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
