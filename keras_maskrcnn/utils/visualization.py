import cv2
import numpy as np
from keras_retinanet.utils.colors import label_color


def draw_mask(image, box, mask, color=None, binarize_threshold=0.5):
    """ Draws a mask in a given box.

    Args
        image              : Three dimensional image to draw on.
        box                : Vector of at least 4 values (x1, y1, x2, y2) representing a box in the image.
        mask               : A 2D float mask which will be reshaped to the size of the box, binarized and drawn over the image.
        color              : Color to draw the mask with. If the box has 5 values, the last value is assumed to be the label and used to construct a default color.
        binarize_threshold : Threshold used for binarizing the mask.
    """
    if color is None:
        if len(box) > 4:
            color = label_color(int(box[4]))
        else:
            color = (0, 255, 0)

    # resize to fit the box
    mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

    # binarize the mask
    mask = (mask > binarize_threshold).astype(np.uint8)

    # draw the mask in the image
    mask_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    mask_image[box[1]:box[3], box[0]:box[2]] = mask
    mask = mask_image

    # compute a nice border around the mask
    border = mask - cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    # apply color to the mask and border
    mask = (np.stack([mask] * 3, axis=2) * color).astype(np.uint8)
    border = (np.stack([border] * 3, axis=2) * (255, 255, 255)).astype(np.uint8)

    # merge the two to get a blended effect
    #mask = cv2.addWeighted(mask, 0.2, eroded_mask + border, 0.8, 0.0)

    # draw the mask
    indices = np.where(mask != [0, 0, 0])
    image[indices[0], indices[1], :] = 0.5 * image[indices[0], indices[1], :] + 0.5 * mask[indices[0], indices[1], :]

    # draw the border
    indices = np.where(border != [0, 0, 0])
    image[indices[0], indices[1], :] = 0.2 * image[indices[0], indices[1], :] + 0.8 * border[indices[0], indices[1], :]


def draw_masks(image, boxes, masks, color=None, binarize_threshold=0.5):
    """ Draws a list of masks given a list of boxes.

    Args
        image              : Three dimensional image to draw on.
        box                : Matrix of shape (N, >=4) (at least 4 values: (x1, y1, x2, y2)) representing boxes in the image.
        mask               : Matrix of shape (N, H, W) of N masks of shape (H, W) which will be reshaped to the size of the corresponding box, binarized and drawn over the image.
        color              : Single color or list of color to draw the masks with.
        binarize_threshold : Threshold used for binarizing the masks.
    """
    for box, mask in zip(boxes, masks):
        draw_mask(image, box, mask, color=color, binarize_threshold=binarize_threshold)
