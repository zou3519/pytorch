import argparse
from ..autograd.utils import YamlLoader, CodeTemplate, write
from ..autograd.gen_python_functions import get_py_torch_functions, get_py_variable_methods
from ..autograd.gen_autograd import load_aten_declarations

# Call via python -mtools.vmap.gen_vmap > gen.out

REG = CodeTemplate("""
m.impl_UNBOXED("${overload_name}", ${fallback_generator}<
  ${func_type},
  static_cast<${func_type}>(${func})${,non_tensor_input_args}
  >);
""")

LAMB = CodeTemplate("""
static constexpr auto tensors_at_front_${operator_name} = [](${adjusted_formals}) -> ${result_type} {
  return self.${operator_name}(${args});
};
""")

def is_method(decl):
    # ['Type', 'namespace' or 'Tensor']
    method_of = decl['method_of']
    return method_of[-1] == 'Tensor'

def get_fallback_generator(num_tensors, is_method):
    assert num_tensors <= 3
    if is_method:
        return 'inplaceMethodFallback' + str(num_tensors)
    return 'inplaceFuncFallback' + str(num_tensors)

def get_func(decl):
    if is_method(decl):
        return '&Tensor::' + decl['operator_name']
    else:
        return 'at::' + decl['operator_name']

def get_input_types_from_formals(formals):
    return [' '.join(formal.split(' ')[:-1]) for formal in formals]

def get_input_types(decl, should_ignore_self):
    input_types = get_input_types_from_formals(decl['formals'])
    if should_ignore_self:
        return input_types[1:]
    return input_types

def get_func_type(decl):
    if is_method(decl):
        base = '{return_types} (Tensor::*)({input_types}) const'
    else:
        base = '{return_types} (*)({input_types})'
    return base.format(
        return_types=decl['return_type'],
        input_types=', '.join(get_input_types(decl, is_method(decl))))

def num_input_tensors(decl):
    formals = decl['formals']
    is_tensor_formal = [1 if 'Tensor' in formal else 0 for formal in formals]
    return sum(is_tensor_formal)

def tensor_inputs_come_first(decl):
    formals = decl['formals']
    is_tensor_formal = [1 if 'Tensor' in formal else 0 for formal in formals]
    no_tensors = False
    for is_tensor in is_tensor_formal:
        if no_tensors and is_tensor:
            return False
        if not is_tensor:
            no_tensors = True
    return True

# TODO: what happens if tensor args aren't at the front of the tensor?
# define a lambda that does a std::forward in the codegen...

def gen_fallback_registration(decl):
    input_tensors = num_input_tensors(decl)

    if not tensor_inputs_come_first(decl):
        formals = decl['formals']
        formals_with_tensors_first = (
            list(filter(lambda formal: 'Tensor' in formal, formals)) +
            list(filter(lambda formal: 'Tensor' not in formal, formals)))
        adjusted_input_types = get_input_types_from_formals(formals_with_tensors_first)

        workaround = LAMB.substitute(
            operator_name=decl['operator_name'],
            adjusted_formals=formals_with_tensors_first,
            result_type=decl['return_type'],
            args=decl['args'][1:])
        res = REG.substitute(
            overload_name=decl['unqual_operator_name_with_overload'],
            fallback_generator=get_fallback_generator(input_tensors, False),
            func='tensors_at_front_' + decl['operator_name'],
            func_type='{return_type} (*)({input_type})'.format(
                return_type=decl['return_type'],
                input_type=', '.join(adjusted_input_types)
            ),
            non_tensor_input_args=adjusted_input_types[input_tensors:])
        return '{\n' + workaround + res + '\n}\n'

    return ''
    # NB: assumes that all of the tensors are at the FRONT of the operator.
    # TODO: this isn't generally true, go fix this.
    res = REG.substitute(
        overload_name=decl['unqual_operator_name_with_overload'],
        fallback_generator=get_fallback_generator(input_tensors, is_method(decl)),
        func=get_func(decl),
        func_type=get_func_type(decl),
        non_tensor_input_args=get_input_types(decl, False)[input_tensors:])
    return res

BLACKLIST = [
    # These operations are not "batch independent"
    "requires_grad_",
    "rename_"
    "set_",
    "embedding_renorm_",  # TODO, check on this, idk what it is
    "resize_",
    "detach_",
    "squeeze_",
    "t_",
    "transpose_",
    "as_strided_",

    # veto sparse
    "sparse_resize_",
    "sparse_resize_and_clear_",
    "copy_sparse_to_sparse_",
]

def should_gen_for(decl):
    operator_name = decl['operator_name']

    # Blacklist private operators
    if operator_name.startswith('_') and not operator_name.startswith('__'):
        return False
    # Blacklist things on the blacklist
    if operator_name in BLACKLIST:
        return False
    return True


def gen_vmap(declarations_path, out):
    declarations = load_aten_declarations(declarations_path)
    declarations = [decl for decl in declarations if should_gen_for(decl)]

    inplace_decls = [decl for decl in declarations if decl['inplace']]

    registrations = [gen_fallback_registration(decl) for decl in inplace_decls]
    registrations = [r for r in registrations if len(r) > 0]
    print('\n'.join(registrations))

def main():
    parser = argparse.ArgumentParser(
        description='Generate batching rules for vmap')
    parser.add_argument('--declarations-path', metavar='DECL',
                        default='torch/share/ATen/Declarations.yaml',
                        help='path to Declarations.yaml')
    parser.add_argument('--out', metavar='OUT',
                        default='.',
                        help='path to output directory')
    args = parser.parse_args()
    gen_vmap(args.declarations_path, args.out)


if __name__ == '__main__':
    main()
