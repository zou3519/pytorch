def get_fake_impl(name):
    def fake_impl(*args, **kwargs):
        raise NotImplementedError('NYI: ' + name)
    return fake_impl

blacklisted_attrs = {
    # Tensor
    '__class__',
    '__getattribute__',
    '__setattr__',

    # torch
    'Tensor',
}


class Registry(object):
    def __init__(self, obj):
        self.ntorch_functions = {}
        self.torch_functions = {}
        self.obj = obj

        for fn in dir(self.obj):
            self.torch_functions[fn] = getattr(self.obj, fn)

    def register(self, fn, name=None):
        if name is None:
            name = fn.__name__
        if name in self.ntorch_functions.keys():
            raise RuntimeError('Attempted to register \'{}\' twice'.format(name))
        self.ntorch_functions[name] = fn

    def set_fns(self, fn_dict, fill_in_missing_fns=False):
        for fn in dir(self.obj):
            # If it is isn't callable, then don't override
            if not hasattr(getattr(self.obj, fn), '__call__'):
                continue

            if fn not in fn_dict.keys() and fill_in_missing_fns:
                if fn in blacklisted_attrs:
                    continue
                setattr(self.obj, fn, get_fake_impl(fn))
            else:
                setattr(self.obj, fn, fn_dict[fn])

        # Add in the new things
        for name, fn in self.ntorch_functions.items():
            if name not in dir(self.obj):
                setattr(self.obj, name, fn)

    def monkey_patch(self):
        self.set_fns(self.ntorch_functions, fill_in_missing_fns=True)

    def undo_monkey_patch(self):
        self.set_fns(self.torch_functions)
