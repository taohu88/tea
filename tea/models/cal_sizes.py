from tea.utils.commons import islist

def conv2d_out_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if not islist(h_w):
        h_w = (h_w, h_w)

    if not islist(kernel_size):
        kernel_size = (kernel_size, kernel_size)

    if not islist(stride):
        stride = (stride, stride)

    if not islist(pad):
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def conv2d_transp_out_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if not islist(h_w):
        h_w = (h_w, h_w)

    if not islist(kernel_size):
        kernel_size = (kernel_size, kernel_size)

    if not islist(stride):
        stride = (stride, stride)

    if not islist(pad):
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w