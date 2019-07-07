def lrelu(x, n, leak=0.2): 
    """
    Leaky Relu function
    
    Arguments: 
    x -- The actual data to be applied into the leaky relu function
    n -- The name of the returned sequence
    leak -- The small amount of gradient applied when a unit is inactive
    
    Returns:
    result -- The end unit after being applied into the leaky relu function
    """
    
    result = tf.maximum(x, leak * x, name=n) 
    return result