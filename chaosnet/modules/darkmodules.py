#
#   Darknet related layers
#   Copyright EAVISE
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Conv2dBatchReLU', 'GlobalAvgPool2d', 'SumLayer', 'ConcatLayer',
           'PaddedMaxPool2d', 'Reorg', 'YOLO3Layer']
log = logging.getLogger(__name__)


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.LeakyReLU`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dBatchReLU(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.LeakyReLU, 0.1, inplace=True)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.01, relu=lambda: nn.LeakyReLU(0.1, inplace = True)):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.momentum = momentum

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
            relu()
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(B, C)
        return x


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class SumLayer(nn.Module):
    """SumLayer used to implement shortcut layer or residual layer"""

    def __init__(self, froms):
        super(SumLayer, self).__init__()
        self.froms = froms

    def forward(self, prev_outputs):
        f = self.froms[0]
        x = prev_outputs[f]
        for f in self.froms[1:]:
            x = x + prev_outputs[f]
        return x

    def __repr__(self):
        s = '{name}({froms})'
        return s.format(name=self.__class__.__name__, froms=self.froms)


class ConcatLayer(nn.Module):
    """ConcatLayer concate outputs from several layers.
    It is used to implement route in darknet
    """

    def __init__(self, froms):
        super(ConcatLayer, self).__init__()
        self.froms = froms

    def forward(self, prev_outputs):
        x = torch.cat([prev_outputs[i] for i in self.froms], 1)
        return x


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super().__init__()
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')
        self.stride = stride
        self.darknet = True

    def extra_repr(self):
        darknet_mode_str = ', darknet_compatible' if self.darknet else ''
        return f'stride={self.stride}{darknet_mode_str}'

    def forward(self, x):
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        if H % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {W} is not divisible by {self.stride}')

        # darknet compatible version from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
        if self.darknet:
            x = x.view(B, C//(self.stride**2), H, self.stride, W, self.stride).contiguous()
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(B, -1, H//self.stride, W//self.stride)
        else:
            ws, hs = self.stride, self.stride
            x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3, 4).contiguous()
            x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2, 3).contiguous()
            x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1, 2).contiguous()
            x = x.view(B, hs*ws*C, H//hs, W//ws)

        return x


class YOLO3Layer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLO3Layer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output
