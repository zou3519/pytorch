class NameChecker:
    def __init__(self, weak=False):
        self.typevars = {}
        self.weak = weak

    def match(self, tensor, *names):
        for name, annotated in zip(tensor.names, names):
            if name is None or annotated is None:
                continue
            if isinstance(annotated, str):
                if not self.weak and name != annotated:
                    raise RuntimeError('Name mismatch: {} and {}'.format(name, annotated))
                continue

            assert isinstance(annotated, TypeVar)
            typ = typevar_name(annotated)
            if typ not in self.typevars.keys():
                self.typevars[typ] = name
                continue

            if name == self.typevars[typ]:
                continue

            if weak:
                self.typevars[typ] = float('NaN')
                continue

            raise RuntimeError(
                ('Name mismatch: {} was previously matched with \'{}\' ' +
                 'but is now also matched with \'{}\'').format(
                     typ, self.typevars[typ], name))

        return self

    def lookup(self, *names):
        result = []
        for name in names:
            if isinstance(name, TypeVar):
                typ = typevar_name(name)
                result.append(self.typevars[typ])
            else:
                result.append(name)
        return tuple(result)
