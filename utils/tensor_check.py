import torch

def is_empty_and_matches(tensor, target_device='cuda:0', target_shape=(1, 0, 2)):
    """
    Check if the tensor is empty and optionally matches the given device and shape.
    
    Args:
    - tensor (torch.Tensor): The tensor to check.
    - target_device (str): The device of the target tensor.
    - target_shape (tuple): The shape of the target tensor.
    
    Returns:
    - bool: True if the tensor is empty and matches the device and shape, False otherwise.
    """
    # Check if the tensor is empty
    is_empty = tensor.nelement() == 0
    
    # Optionally, check if the device and shape match
    matches_device = tensor.device == torch.device(target_device)
    matches_shape = tensor.shape == target_shape
    
    return is_empty and matches_device and matches_shape
