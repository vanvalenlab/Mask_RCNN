if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import cv2
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class CellDataset(utils.Dataset):

    def load_trainingcells(self, dataset_dir):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("cell", 1, "cell")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        #dataset_dir = os.path.join(dataset_dir, subset_dir)
        image_ids = next(os.walk(dataset_dir))[2]
        #To suffle the images just to make sure that any bais due to 
        #generation does not kick in.
        image_ids = list(set(image_ids))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "cell",
                image_id=image_id,
                path=os.path.join(dataset_dir,image_id))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        image_name=info['path'].split("/")[-1]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        #Get the exact full mask for that image
        for f in next(os.walk(mask_dir))[2]:
        	if f.endswith(str(image_name)):
        		full_mask = cv2.imread(os.path.join(mask_dir, f),0)

        #extract indivial masks of cells from the full mask
        lb = label(full_mask)

		msks = []
		for key in range(1,lb[1]+1):
		    newim=np.zeros(full_mask.shape)
		    for i in range(full_mask.shape[0]):
		        for j in range(full_mask.shape[1]):
		            if lb[0][i][j]==key:
		                newim[i][j]=1
		    msks.append(newim)
		msks=np.astype(np.bool)

        # Combine these masks of indiviual cells
        mask = np.stack(msks, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
