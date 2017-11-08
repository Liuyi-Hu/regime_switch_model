# Part of the code are from Python package 'hmmlearn'
# Their source code can be found here: https://github.com/hmmlearn/hmmlearn



def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1

    Parameters
    ----------
    a: numpy array
        Non-normalized input data

    axis: int
        Dimension along which normalization is performed

    Notes
    -----
    Modifies the input **inplace**
    """
    #a = np.array(a, dtype=float)
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape
    a /= a_sum

