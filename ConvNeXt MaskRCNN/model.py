from torchvision.models.detection import MaskRCNN
from .dataset import BuildDataset
from torch.utils.data import DataLoader
from .inference import Config
import torch
from .utils import get_file_dir, convnext_fpn_backbone, Trainer

train_img_path = './cell_tracking_challenge/train/images'
train_mask_path = './cell_tracking_challenge/train/masks'
test_img_path = './cell_tracking_challenge/test/images'
test_mask_path = './cell_tracking_challenge/test/masks'

train_dataset = BuildDataset(get_file_dir(train_img_path), get_file_dir(train_mask_path))
test_dataset = BuildDataset(get_file_dir(test_img_path), get_file_dir(test_mask_path))

train_loader = DataLoader(train_dataset, batch_size=Config.train_bs, 
                            num_workers=4, shuffle=True, pin_memory=True, 
                          drop_last=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=Config.val_bs, 
                            num_workers=4, shuffle=True, pin_memory=True, 
                         collate_fn=lambda x: tuple(zip(*x)))

backbone = convnext_fpn_backbone(
    Config.backbone,
    Config.trainable_layers
)

model = MaskRCNN(
    backbone, 
    num_classes=Config.num_classes, 
    max_size=Config.max_size,
    min_size=Config.min_size,
)
if Config.pretrained:
    try:
        model.load_state_dict(torch.load(Config.weights, map_location=Config.device))
    except:
        ValueError(f"You did not enter the model path correctly, got {Config.weights}. Enter a String type path")

model.to(Config.device)

params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=Config.lr, momentum=Config.momentum, weight_decay=Config.weight_decay)
optimizer = torch.optim.AdamW(
    params, 
    lr=Config.lr, 
    weight_decay=Config.weight_decay
)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=Config.lr_steps, gamma=Config.lr_gamma)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=Config.step_size,
    gamma=Config.gamma
)
scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(
    optimizer=optimizer,
    max_epochs=Config.epochs,
    device=Config.device,
    scaler=scaler,
    verbose_num=Config.verbose_num,
    split_size=Config.split_size,
    val_bs=Config.val_bs
)

history = trainer.fit(
    model, 
    train_dataloader=train_loader, 
    ckpt_path=Config.save_path + Config.model_name + ".pth"
)
    