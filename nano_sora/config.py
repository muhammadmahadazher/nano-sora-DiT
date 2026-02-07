class Config:
    # Data
    input_size = 32
    patch_size = 2
    in_channels = 3
    num_classes = 10
    
    # Model
    hidden_size = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    class_dropout_prob = 0.1
    
    # Training
    batch_size = 64
    lr = 3e-4
    epochs = 100
    checkpoint_interval = 10
    mixed_precision = "fp16"
    
    # Paths
    checkpoint_dir = "checkpoints"
