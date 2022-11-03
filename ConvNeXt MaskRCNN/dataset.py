from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class BuildDataset(Dataset):
	"""
	Run once when instantiating the Dataset object.
	You can customize your Dataset inherited class.
	Parameters
	----------
	img_paths : list
		List of image file paths.
	mask_paths (list) : Defaults to None
		List of ground truth file paths.
        """
    def __init__(self, 
                 img_paths, 
                 mask_paths=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        
    def __len__(self):
        """
        Returns the number of samples in our dataset.
        Returns
        -------    
        num_datas : int    
            Number of datas.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at 
        the given index idx. Based on the index, it 
        identifies the image’s location on disk, 
        converts that to a tensor using imageio and torch.Tensor, 
        retrieves the corresponding label from the 
        ground truth data in self.mask_paths, calls the transform 
        functions on them (if applicable), and returns 
        the tensor image and corresponding label in a tuple.
        Parameters
        ----------
        idx : int
            It is given to load the image or image and target 
            according to the index from the Dataset object.
        Returns
        -------   
        img, target : torch.Tensor, dictionary
            The transformed image and its corresponding 
            targets. If the mask path is None, it 
            will only return the transformed image.
            output_shape_img: (UInt8Tensor[N, 3, H, W])
            target dictionary includes:
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in 
                  [xmin, ymin, xmax, ymax] format, with values between 0-H and 0-W
                - labels (Int64Tensor[N]): the class label for each 
                  ground-truth box
                - masks (UInt8Tensor[N, H, W]): the segmentation binary 
                  masks for each instance
                - image_id (Int64Tensor[N]): the index of the image in the dataset.
                - area (Float): the area occupied by the bboxes in the image.
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
        if self.mask_paths is not None:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]

            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area

            return img, target
        else:
            return img