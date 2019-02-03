from ._bricks import _ModuleEnum

def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1][_ModuleEnum.type] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][_ModuleEnum(key.rstrip())] = value.strip()

    return module_defs
