# Tea

Tea is a deep learning application library to make it easy to test ideas quick, to build model from end to end.
It is config driven so users write minimal code to try ideas.

It won't be another deep learning framework such as pytorch, tensorflow. 

It is easy to make things complicated, but it is hard to make things simple. As the old saying goes “大道至简”.
Thus, it strives the best to make code understandable and maintainable even if we have to make code longer.
It avoids unnecessary tricks whenever possible.

# Quick to build model from config file
The following is an example on how modified vgg can be built from config file.

```
[input]
# It is super important to get input dimension right
# Batch x channel x H x W
size=None x 3 x 56 x 56

#
# M --> Maxpool
#
[quickconvs-0]
filters=64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512
kernel=3
stride=1
has_pad=1
has_bn=1
pool_kernel=2
pool_stride=2
activation=relu

[quickfcs-1]
reshape=1
size=4096, D, 2048, D
activation=relu

[fc-3]
size=200
activation=linear
```
