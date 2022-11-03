import torch

class Config:
    """
    MaskRCNN model configuration class. 
    Enter specific configurations for the own model.
    """
    seed              = 101
    model_name        = 'MaskRCNN'
    backbone          = 'convnext_tiny'
    save_path         = '/content/'
	class_names       = ['__background__', 'cell']
    weights           = None # file path with weights of your pre-trained model
    pretrained        = False # If you have a pre-trained model, load weights
    max_size          = 512
    min_size          = 512
    trainable_layers  = 5
    train_bs          = 1 # train batch size
    val_bs            = 4 # validation batch size
    split_size        = 0.2 # If you do not have a validation dataset, enter the split size between 0 and 1
    epochs            = 30 
    lr                = 1e-5 
    momentum          = 0.9
    min_lr            = 1e-8
    step_size         = 3
    gamma             = 0.1
    T_max             = int(100*6*1.8)
    T_0               = 25
    weight_decay      = 5e-6
    n_accumulate      = 32//train_bs
    n_fold            = 5
    num_classes       = 2
    device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    verbose_num       = 10