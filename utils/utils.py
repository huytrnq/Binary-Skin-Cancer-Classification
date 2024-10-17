

def sliding_window(image, window_size, step_size):
    """
    A generator that yields the coordinates and the window of the image.
    
    Args:
        image (numpy array): The input image (grayscale or color).
        window_size (tuple): The size of the window (height, width).
        step_size (int): The step size for sliding the window.
        
    Yields:
        (x, y, window): The top-left corner (x, y) and the window (region) of the image.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])