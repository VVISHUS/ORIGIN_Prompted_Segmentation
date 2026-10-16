MODELS = {}


def register(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def build_model(cfg_model):
    kwargs = dict(cfg_model)
    name = kwargs.pop("name")
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
