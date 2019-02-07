import ignite.metrics  as ig


class Accuracy(ig.Accuracy):

    # there is a bug in accuracy, which specific need two inputs
    def __init__(self, output_transform=lambda x: x[:2]):
        super().__init__(output_transform)
