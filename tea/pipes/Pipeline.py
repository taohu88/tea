# Some basic pipeline classes


class SequencePipeline():
    """
    A sequence pipeline to link add pipes together
    Pipe is nothing but a callable, which only takes either positional or keyed params
    """
    def __init__(self, pipes):
        self.pipes = pipes

    def add_pipe(self, pipe):
        self.pipes.append[pipe]
        return self

    def __call__(self, **kwargs):
        ins = kwargs
        for p in self.pipes:
            if isinstance(ins, dict):
                ins = p(**ins)
            elif isinstance(ins, list):
                ins = p(*ins)
            elif isinstance(ins, tuple):
                ins = p(*ins)
            else:# assume scale type
                ins = p(ins)

        return ins

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.pipes:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
