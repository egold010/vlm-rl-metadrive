from metadrive.constants import Semantics
import numpy as np

def get_class_names():
    return [name for name in Semantics.__dict__.keys() if name.isupper()]

def get_num_classes():
    """
    Returns the number of classes in the semantic map.
    """
    class_names = [name for name in Semantics.__dict__.keys() if name.isupper()]
    a = Semantics.__dict__.keys()

    return len(class_names)

def one_hot_encode_semantic_map(img_array):
    """
    Converts an (H, W, 3) RGB semantic map array into a one-hot encoded
    array of shape (num_classes, H, W).

    Parameters:
    - img_array: np.ndarray of shape (H, W, 3)

    Returns:
    - one_hot: np.ndarray of shape (num_classes, H, W), dtype uint8.
    - class_names: List of class names in the same order as channels.
    """
    # Build color map from Semantics class
    class_names = [name for name in Semantics.__dict__.keys() if name.isupper()]
    color_map = {name: Semantics.__dict__[name].color for name in class_names}
    H, W, C = img_array.shape
    assert C == 3, "Input must be (H, W, 3)"
    
    class_names = list(color_map.keys())
    num_classes = len(class_names)
    one_hot = np.zeros((num_classes, H, W), dtype=np.uint8)

    for idx, cls in enumerate(class_names):
        color = color_map[cls]
        mask = np.all(img_array == color, axis=-1)
        one_hot[idx, mask] = 1

    return one_hot