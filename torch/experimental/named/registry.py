def get_fake_impl(name):
    def fake_impl(*args, **kwargs):
        raise NotImplementedError('NYI: ' + name)
    return fake_impl


class Registry(object):
    def __init__(self, obj):
        self.ntorch_functions = {}
        self.torch_functions = {}
        self.obj = obj

        for fn in dir(self.obj):
            self.torch_functions[fn] = getattr(self.obj, fn)

    def register(self, fn):
        name = fn.__name__
        assert name not in self.ntorch_functions.keys()
        self.ntorch_functions[name] = fn

    def set_fns(self, fn_dict, fill_in_missing_fns=False):
        for fn in dir(torch):
            if fn not in fn_dict.keys() and fill_in_missing_fns:
                if fn == '__name__':
                    continue
                setattr(self.obj, fn, get_fake_impl(fn))
            else:
                setattr(self.obj, fn, fn_dict[fn])

    def monkey_patch(self):
        self.set_fns(self.ntorch_functions, fill_in_missing_fns=True)

    def undo_monkey_patch(self):
        self.set_fns(self.torch_functions)
