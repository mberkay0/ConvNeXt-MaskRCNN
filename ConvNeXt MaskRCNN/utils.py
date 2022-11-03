import os
import glob
import torch
import random
import numpy as np
import re
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
from .inference import Config
from torch import nn
import torch.distributed as dist
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, convnext_small, ConvNeXt_Small_Weights, convnext_large, ConvNeXt_Large_Weights
import datetime
import errno
import time
import sys
from collections import defaultdict, deque
import math
from torch.utils.data import Dataset, DataLoader, Subset


CLASS_NAMES = Config.class_names

def set_seed(seed = 42):
    """
    Sets the seed of the entire notebook 
    so results are the same every time we run.
    This is for REPRODUCIBILITY.
    Parameters
    ----------
    seed : int
        This is the seed for the next random number. 
        If omitted, then it takes system time to 
        generate next random number.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE!')
    
def atoi(text):
    return int(text) if text.isdigit() else text
    
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]

def get_file_dir(directory, get_dirs=True):
    """
    Creates a new folder or returns files 
    in the entered directory.
    Parameters
    ----------
    directory : string
        File path.
    get_dirs (bool, optional): Defaults to True.
        If true, it returns the path of all 
        files in the given directory.
    Returns
    ----------
    directories: list
        All files in the directory.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Failed to create file!')

    if get_dirs:
        for roots,dirs,files in os.walk(directory):               
            if files:
                directories = [roots + os.sep + file for file in  files]
                directories.sort(key=natural_keys)

        return directories

def plot_batch(dataset, size=5, fontsize=15, alpha=0.3):
    """
    Creates a new folder or returns files 
    in the entered directory.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset object
        Given Dataset object.
    size (int, optional): Defaults to 5.
        Enter how many examples you want to visualize.
    fontsize (int, optional): Defaults to 15.
        Font size of each image caption.
    alpha (float, optional): Defaults to 0.3.
        Enter the visibility of the mask to be overlaid on 
        the original image. It must be between 0 and 1.
    """
    for idx in range(size):
        image, target = dataset[idx]
        image = np.array(image.permute((1, 2, 0))).astype(np.uint8)
        bboxes = target['boxes']
        masks = target['masks']
        expanded_mask = np.zeros(masks[0].shape[:2])
        for i in range(len(masks)):
            expanded_mask[np.where(masks[i] == 1)] = i + 1
            cv2.rectangle(image, (
                int(bboxes[i][0]), 
                int(bboxes[i][1])), 
                (int(bboxes[i][2]), 
                 int(bboxes[i][3])
            ), 
            color=(0, 255, 0), 
            thickness=2)

        plt.figure(figsize=(4 * 3, 5))

        plt.subplot(1, 3, 1); plt.imshow(image)
        plt.title('image', fontsize=fontsize)
        plt.axis('OFF')

        plt.subplot(1, 3, 2); plt.imshow(np.squeeze(expanded_mask))
        plt.title('mask', fontsize=fontsize)
        plt.axis('OFF')

        plt.subplot(1, 3, 3); plt.imshow(image); plt.imshow(np.squeeze(expanded_mask), alpha=alpha)
        plt.title('overlay', fontsize=fontsize)
        plt.axis('OFF')
        
        plt.tight_layout()
        plt.show()

    
def random_color_generator(seed):
    """
    Code that generates random color according 
    to the seed given inside.
    Parameters
    ----------
    seed : int
        Enter an integer to set NumPy's random seed.
    Returns
    ----------
    color: tuple 
        Random generated RGB colors.
    """
    np.random.seed(seed)
    color = np.random.choice(range(255), size=3)
    return (int(color[0]), int(color[1]), int(color[2]))


def get_prediction(model, img_path, confidence, mask_th=0.5):
    """
    It reads the image from the given file path and 
    returns the output predicted by the model.
    Parameters
    ----------
    model : pytorch model
        The model to generate the predictions.
    img_path : str
        The file path with the image to be predicted.
    confidence : float
        Enter a value between 0 and 1 to access reliable results. 
        A value close to 1 is the result that the model predicted 
        with higher confidence.
    mask_th (float, optional): Defaults to 0.5.
        Enter a value between 0 and 1 to threshold the probability 
        map generated by the model.
    Returns
    ----------
    masks : np.uint8
        Thresholded masks according to the given threshold value.
    pred_boxes : list
        Bounding boxes predicted by the model.
    pred_class : list
        Name of predicted classes.
    """
    img = Image.open(img_path).convert('RGB')
    img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
    img = img.to(Config.device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    indices = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>mask_th).squeeze().detach().cpu().numpy()
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:indices+1]
    pred_boxes = pred_boxes[:indices+1]
    pred_class = pred_class[:indices+1]
    return masks, pred_boxes, pred_class

def visualize(model, img_path, confidence=0.5, rect_thickness=2, 
              text_size=1.4, text_thickness=2):
    """
    It reads the image from the given file path and 
    returns the output predicted by the model.
    Parameters
    ----------
    model : pytorch model
        The model to generate the predictions.
    img_path : str
        The file path with the image to be predicted.
    confidence (float, optional): Defaults to 0.5.
        Enter a value between 0 and 1 to access reliable results. 
        A value close to 1 is the result that the model predicted 
        with higher confidence.
    rect_thickness (int, optional): Defaults to 2.
        Bounding box thickness.
    text_size (float, optional): Defaults to 1.4.
        Font size of class label to overwrite bounding boxes.
    text_thickness (int, optional): Defaults to 2.
        Font thickness of class label to overwrite bounding boxes.
    """
    masks, boxes, pred_cls = get_prediction(model, img_path, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i, mask in enumerate(masks):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        r[mask == 1], g[mask == 1], b[mask == 1] = random_color_generator(i)
        rgb_mask = np.stack([r, g, b], axis=2)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, 
                      (int(boxes[i][0][0]), int(boxes[i][0][1])), 
                      (int(boxes[i][1][0]), int(boxes[i][1][1])), 
                      color=(0, 255, 0), thickness=rect_thickness)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), 
                    cv2.FONT_HERSHEY_PLAIN, text_size, (0,255,0),thickness=text_thickness)
    plt.figure(figsize=(10,20))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Parameters
    ----------
    data: any picklable object.
    Returns
    ------- 
    list[data]: list of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    Parameters
    ----------
    input_dict (dict): all the values will be reduced.
    average (bool): whether to do average or sum.
    Returns
    ------- 
    reduced_dict (dict): reduced results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
	
class DiceScore:
    """
    The Sørensen–Dice coefficient is a statistic used to gauge 
    the similarity of two samples. A metric class for Pytorch that 
    measures the dice coefficient. Each of the scores is between 0 and 1. 
    Scores close to 1 indicate higher confidence in the ground truth.
    For more information about that dice coefficient:
    https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
    Parameters
    ----------
    thresh (float): defaults to 0.5.
        The value to be used to threshold the given probability
        map between 0 and 1.
    smooth (float): defaults to 1e-8.
        Smoothness factor for division by the zero.
    """
    def __init__(self, thresh=0.5, smooth=1e-8):
        self.thresh = thresh
        self.smooth = smooth

    
    def __call__(self, pred, ground_truth):
        return self.dice_score(pred, ground_truth)

    
    def __name__(self):
        return "dice_score"


    def _merge_masks(self, masks):
        """
        Merge a same size masks.
        Parameters
        ----------
        masks : ndarray
            Binary segmented images.
        Returns
        -------
        merged_mask : ndarray
            Merged binary segmented image.
        """
        merged_mask = torch.zeros_like(masks[0])
        for mask in masks:
            merged_mask += mask
        return merged_mask

    def dice_score(self, pred, ground_truth):
        """
        Calculates the dice coefficient between two given tensors.
        Parameters
        ----------
        pred : torch.Tensor.float
            The probability map predicted by the model.
        ground_truth : torch.Tensor.float
            Ground truth same size with pred.
        Returns
        -------
        score : torch.float
            Returns the calculated dice coefficient.
        """
        if len(pred) >= 1 and len(ground_truth) >= 1:
            pred = self._merge_masks(pred)
            ground_truth = self._merge_masks(ground_truth)
        pred = (pred >= self.thresh).float()
        score = (2 * (pred * ground_truth).sum()) / ((pred + ground_truth).sum() + self.smooth)
        return score


class SaveBestModelOnly:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    Parameters
    ----------
    best_valid_loss float: defaults to float('inf'). 
        It keeps the best validation loss.       
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, ckpt_path, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        """
        Parameters
        ----------
        ckpt_path : str
            File path to save the model.
        current_valid_loss : float
            Loss of validation step in current epoch.
        epoch : int
            Current epoch number.
        model : torch.nn.Module
            Trained pytorch model.
        optimizer : torch.optim
            Current model optimizer.
        criterion : float
            Loss of training in current epoch.
        """
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, ckpt_path)
				
				
class Trainer:
    """
    Trainer is a simple but feature-complete training and eval 
    loop for PyTorch, optimized for torchvision.models.MaskRCNN.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        To use in training step torch.optim you have to construct an 
        optimizer object that will hold the current state and update 
        the parameters based on the computed gradients.
    max_epochs : int
        It is the maximum number of epochs for training the model.
    device : str    
        Enables you to specify the device type responsible to load a 
        tensor into memory.
    verbose_num (int): defaults to 10.
        Specifies how often to print information during training.
        If you have 100 pieces of data, it indicates will print 
        your loss values every 10 data.
    split_size (float): defaults to 0.2.
        If float, should be between 0.0 and 1.0 and represent the 
        proportion of the dataset to include in the validation split.
    val_bs (int): defaults to 4.
        If validation data is not available, it must be defined. 
        The batch size for the validation step.
    lr_scheduler (torch.optim.lr_scheduler): defaults to None.
        Pytorch learning rate scheduler is used to find the optimal 
        learning rate for various models by conisdering the model 
        architecture and parameters.
    scaler (torch.cuda.amp.scaler): defaults to None.
        An instance ``scaler`` helps perform the steps of gradient #
        scaling conveniently.
    Attributes
    ----------
    history : defaultdict(list)
        This dictionary keeps training and validation losses 
        achieved during each epoch.
    """
    def __init__(self, optimizer, max_epochs, device, 
                 verbose_num=10, split_size=0.2, val_bs=4,
                 lr_scheduler=None, scaler=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.device = device
        self.print_freq = verbose_num
        self.scaler = scaler
        self.split_size = split_size
        self.val_bs = val_bs
        self.save_ckpt = SaveBestModelOnly()
        self.history = defaultdict(list)


    def fit(self, model, train_dataloader, 
            val_dataloader=None, ckpt_path=None):
        """
        Parameters
        ----------
        model : ([`PreTrainedModel`] or `torch.nn.Module`, *optional*)
            The model to train, evaluate or use for predictions. 
            If not provided, a `model_init` must be passed.
        train_dataloader : (torch.utils.data.DataLoader)
            Data loader. Combines a dataset and a sampler, and provides 
            an iterable over the given dataset for training.
        val_dataloader (torch.utils.data.DataLoader): defaults to None.
            Data loader. Combines a dataset and a sampler, and provides 
            an iterable over the given dataset for validation step. 
            If set to None, the Trainer class splits the validation set 
            randomly according to the entered split size amount.
        ckpt_path (str): defaults to None.
            The path to the file is required to save the best version 
            of the model during training. If it is None, model can not save.
        Returns
        -------
        history : defaultdict(list)
            This dictionary keeps training and validation losses 
            achieved during each epoch.
        """
        
        if val_dataloader is None:
            train_dataloader, val_dataloader = self._split_train_val(train_dataloader)
        
        print('----------------------------------------------')
        print('\t\tStart training')
        print('----------------------------------------------')
        
        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            # train
            self._train(model, train_dataloader, epoch)
            
            # validate after every epoch
            self._validate(model, val_dataloader, epoch)

            # self.lr_scheduler.step()
            if ckpt_path is not None:
                self.save_ckpt(
                    ckpt_path,
                    self.history["val_loss"][epoch-1],
                    epoch,
                    model,
                    self.optimizer,
                    self.history["train_loss"][epoch-1]
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

        return self.history


    def _split_train_val(self, train_dataloader):
        train_dataset = train_dataloader.dataset
        data_size = len(train_dataset)
        train_indices = list(np.random.choice(
            range(data_size), 
            int((1-self.split_size)*data_size), 
            replace=False
        ))

        val_indices = [ind for ind in range(data_size) if ind not in train_indices]

        train_subset = Subset(train_dataset, train_indices)
        val_set = Subset(train_dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=train_dataloader.batch_size,
            shuffle=True, num_workers=train_dataloader.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.val_bs,
            shuffle=True, num_workers=train_dataloader.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        return train_loader, val_loader
            

    def _train(self, model, data_loader, epoch):
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        train_loss = []

        if self.lr_scheduler is None:
            if epoch == 1:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(data_loader) - 1)

                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )

        for images, targets in metric_logger.log_every(data_loader, self.print_freq, header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            train_loss.append(loss_value)

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        
        self.history["train_loss"].append(np.mean(train_loss))

        return metric_logger


    @torch.inference_mode()
    def _validate(self, model, data_loader, epoch):
        print("Validation step:")
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"
        val_loss = []

        for images, targets in metric_logger.log_every(data_loader, len(data_loader), header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            val_loss.append(loss_value)

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        self.history["val_loss"].append(np.mean(val_loss))

        return metric_logger


    @torch.inference_mode()
    def evaluate(self, model, data_loader, metric):
        """
        Parameters
        ----------
        model : ([`PreTrainedModel`] or `torch.nn.Module`, *optional*)
            The model to train, evaluate or use for predictions. 
            If not provided, a `model_init` must be passed.
        data_loader : (torch.utils.data.DataLoader)
            Data loader. Combines a dataset and a sampler, and provides 
            an iterable over the given dataset for evaluation.
        metric (callable): function or class.
            The metric required to measure model evaluation performance.
        """
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = "Model Evaluation:"

        for images, targets in metric_logger.log_every(data_loader, 1, header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            output_masks = outputs[0]['masks']
            target_masks = targets[0]["masks"]

            score = metric(output_masks, target_masks)

            if metric.__name__() == "dice_score":
                metric_logger.update(dice_score=score)
            elif metric.__name__() == "iou_score":
                metric_logger.update(iou_score=score)
            else:
                metric_logger.update(score=score)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
		
		
input_channels_dict = {
    'convnext_tiny': [96, 192, 384, 768],
    'convnext_small': [96, 192, 384, 768],
    'convnext_base': [128, 256, 512, 1024],
    'convnext_large': [192, 384, 768, 1536],
    'convnext_xlarge': [256, 512, 1024, 2048],
}


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models.feature_extraction.create_feature_extractor 
    to extract a submodel that returns the feature maps specified in given backbone 
    feature extractor model.
    Parameters
    ----------
    backbone (nn.Module): Feature extractor ConvNeXt pretrained model.        
    in_channels_list (List[int]): Number of channels for each feature map
        that is returned, in the order they are present in the OrderedDict
    out_channels (int): number of channels in the FPN.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use. 
    extra_blocks (callable, optional): Default None.
        Extra optional FPN blocks.
    Attributes
    ----------
    out_channels : int
        The number of channels in the FPN.
    """

    def __init__(
        self,
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks = None,
        norm_layer = None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def convnext_fpn_backbone(
    backbone_name,
    trainable_layers,
    extra_blocks = None,
    norm_layer = None,
    feature_dict = {'1': '0', '3': '1', '5':'2', '7':'3'},
    out_channels = 256
):
    """
    Returns an FPN-extended backbone network using a feature extractor 
    based on models developed in the article 'A ConvNet for the 2020s'.
    For detailed information about the feature extractor ConvNeXt, read the article.
    https://arxiv.org/abs/2201.03545
    Parameters
    ----------
    backbone_name : str
        ConvNeXt architecture. Possible values are 'convnext_tiny', 'convnext_small', 
        'convnext_base' or 'convnext_large'.
    trainable_layers : int
        Number of trainable (not frozen) layers starting from final block.
        Valid values are between 0 and 8, with 8 meaning all backbone layers 
        are trainable.
    extra_blocks (ExtraFPNBlock or None): default a ``LastLevelMaxPool`` is used.
        If provided, extra operations will be performed. It is expected to take 
        the fpn features, the original features and the names of the original 
        features as input, and returns a new list of feature maps and their 
        corresponding names.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use. It is recommended to use 
        the default value. For details visit: 
        (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267) 
    feature_dict : dictionary
        Contains the names of the 'nn.Sequential' object used in the ConvNeXt 
        model configuration if you need more detailed information, 
        https://github.com/facebookresearch/ConvNeXt. 
    out_channels (int): defaults to 256.
        Number of channels in the FPN.
    Returns
    -------
    BackboneWithFPN : torch.nn.Module
        Returns a specified ConvNeXt backbone with FPN on top. 
        Freezes the specified number of layers in the backbone.
    """
    if backbone_name == "convnext_tiny":
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_small":
        backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_base":
        backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_large":
        backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    else:
        raise ValueError(f"Backbone names should be in the {list(input_channels_dict.keys())}, got {backbone_name}")

    in_channels_list = input_channels_dict[backbone_name]

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(f"Trainable layers should be in the range [0,8], got {trainable_layers}")
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1"][:trainable_layers]
    if trainable_layers == 8:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(
        backbone, in_channels_list, out_channels, 
        extra_blocks=extra_blocks, norm_layer=norm_layer
    )