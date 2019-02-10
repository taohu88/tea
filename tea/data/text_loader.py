# coding: utf-8


class TextLoader:

    def __init__(self, src_iter):
        self.src_iter = src_iter

    def __len__(self):
        return len(self.src_iter)

    def __iter__(self):
        for batch in self.src_iter:
            yield batch.text, batch.target
        return