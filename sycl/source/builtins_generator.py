import itertools
import sys
import os

class TemplatedType:
  def __init__(self, valid_types, valid_sizes):
    self.valid_types = valid_types
    self.valid_sizes = valid_sizes

class Vec(TemplatedType):
  def __init__(self, valid_types, valid_sizes = {1,2,3,4,8,16},
               deprecation_message=None):
    super().__init__(valid_types, valid_sizes)
    self.deprecation_message = deprecation_message

  def get_requirements(self, type_name):
    valid_type_str = ', '.join(self.valid_types)
    valid_sizes_str = ', '.join(map(str, self.valid_sizes))
    return [f'detail::is_vec_or_swizzle_v<{type_name}>',
            f'detail::is_valid_elem_type_v<{type_name}, {valid_type_str}>',
            f'detail::is_valid_size_v<{type_name}, {valid_sizes_str}>']

class Marray(TemplatedType):
  def __init__(self, valid_types, valid_sizes = None):
    super().__init__(valid_types, valid_sizes)

  def get_requirements(self, type_name):
    valid_type_str = ', '.join(self.valid_types)
    result = [f'detail::is_marray_v<{type_name}>',
              f'detail::is_valid_elem_type_v<{type_name}, {valid_type_str}>']
    if self.valid_sizes:
      valid_sizes_str = ', '.join(map(str, self.valid_sizes))
      result.append(f'detail::is_valid_size_v<{type_name}, {valid_sizes_str}>')
    return result

class InstantiatedTemplatedArg:
  def __init__(self, template_name, templated_type):
    self.template_name = template_name
    self.templated_type = templated_type

  def get_requirements(self):
    return self.templated_type.get_requirements(self.template_name)

  def __str__(self):
    return self.template_name

class InstantiatedTemplatedReturnType:
  def __init__(self, related_arg_type, templated_type):
    self.related_arg_type = related_arg_type
    self.templated_type = templated_type

  def __str__(self):
    if isinstance(self.templated_type, Vec):
      # Vectors need conversion as their related type may be a swizzle.
      return f'detail::get_vec_t<{self.related_arg_type}>'
    return str(self.related_arg_type)

class MultiPtr:
  def __init__(self, element_type):
    self.element_type = element_type

  def __str__(self):
    return f'multi_ptr<{self.element_type}, Space, IsDecorated>'

# TODO: Raw pointer variants kept for compatibility. They should be deprecated.
class RawPtr:
  def __init__(self, element_type):
    self.element_type = element_type
    self.deprecation_message = "SYCL builtin functions with raw pointer arguments have been deprecated. Please use multi_ptr."

  def __str__(self):
    return f'{self.element_type}*'

class ElementType:
  def __init__(self, parent_idx):
    self.parent_idx = parent_idx

class InstantiatedElementType:
  def __init__(self, referenced_type, parent_idx):
    self.referenced_type = referenced_type
    self.parent_idx = parent_idx

  def __str__(self):
    if (isinstance(self.referenced_type, InstantiatedTemplatedArg) or
        isinstance(self.referenced_type, InstantiatedTemplatedReturnType)):
      return f'detail::get_elem_type_t<{self.referenced_type}>'
    return self.referenced_type

class ConversionTraitType:
  def __init__(self, trait, parent_idx):
    self.trait = trait
    self.parent_idx = parent_idx

class InstantiatedConversionTraitType:
  def __init__(self, parent_type, trait, parent_idx):
    self.parent_type = parent_type
    self.trait = trait
    self.parent_idx = parent_idx

  def __str__(self):
    return f'detail::{self.trait}<{self.parent_type}>'

### GENTYPE DEFINITIONS
# NOTE: Marray is currently explicitly defined.

floatn = [Vec(["float"])]
vfloatn = [Vec(["float"])]
vfloat3or4 = [Vec(["float"], {3,4})]
mfloatn = [Marray(["float"])]
mfloat3or4 = [Marray(["float"], {3,4})]
genfloatf = ["float", Vec(["float"]), Marray(["float"])]

doublen = [Vec(["double"])]
vdoublen = [Vec(["double"])]
vdouble3or4 = [Vec(["double"], {3,4})]
mdoublen = [Marray(["double"])]
mdouble3or4 = [Marray(["double"], {3,4})]
genfloatd = ["double", Vec(["double"])]

halfn = [Vec(["half"])]
vhalfn = [Vec(["half"])]
vhalf3or4 = [Vec(["half"], {3,4})]
mhalfn = [Marray(["half"])]
mhalf3or4 = [Marray(["half"], {3,4})]
genfloath = ["half", Vec(["half"])]

genfloat = ["float", "double", "half", Vec(["float", "double", "half"]),
            Marray(["float", "double", "half"])]
vgenfloat = [Vec(["float", "double", "half"])]
sgenfloat = ["float", "double", "half"]
mgenfloat = [Marray(["float", "double", "half"])]

vgeofloat = [Vec(["float"], {1,2,3,4})]
vgeodouble = [Vec(["double"], {1,2,3,4})]
vgeohalf = [Vec(["half"], {1,2,3,4})]
mgeofloat = [Marray(["float"], {1,2,3,4})]
mgeodouble = [Marray(["double"], {1,2,3,4})]
mgeohalf = [Marray(["half"], {1,2,3,4})]
gengeofloat = ["float", Vec(["float"], {1,2,3,4}), Marray(["float"], {1,2,3,4})]
gengeodouble = ["double", Vec(["double"], {1,2,3,4}), Marray(["double"], {1,2,3,4})]
gengeohalf = ["half", Vec(["half"], {1,2,3,4}), Marray(["half"], {1,2,3,4})]

vint8n = [Vec(["int8_t"])]
vint16n = [Vec(["int16_t"])]
vint32n = [Vec(["int32_t"])]
vint64n = [Vec(["int64_t"])]
vuint8n = [Vec(["uint8_t"])]
vuint16n = [Vec(["uint16_t"])]
vuint32n = [Vec(["uint32_t"])]
vuint64n = [Vec(["uint64_t"])]
vint64n_ext = [Vec(["int64_t", "long long"])]
vuint64n_ext = [Vec(["uint64_t", "unsigned long long"])]

mint8n = [Marray(["int8_t"])]
mint16n = [Marray(["int16_t"])]
mint32n = [Marray(["int32_t"])]
mint64n = [Marray(["int64_t"])]
muint8n = [Marray(["uint8_t"])]
muint16n = [Marray(["uint16_t"])]
muint32n = [Marray(["uint32_t"])]
muint64n = [Marray(["uint64_t"])]
mcharn = [Marray(["char"])]
mshortn = [Marray(["short"])]
mintn = [Marray(["int"])]
mushortn = [Marray(["unsigned short"])]
muintn = [Marray(["unsigned int"])]
mulongn = [Marray(["unsigned long", "unsigned long long"])]
mbooln = [Marray(["bool"])]

geninteger = ["char", "signed char", "short", "int", "long", "long long",
              "unsigned char", "unsigned short", "unsigned int",
              "unsigned long", "unsigned long long",
              Vec(["int8_t", "int16_t", "int32_t", "int64_t", "uint8_t",
                  "uint16_t","uint32_t","uint64_t", "unsigned long long"]), # unsigned long long non-standard. Deprecated.
              Marray(["char", "signed char", "short", "int", "long", "long long",
                      "unsigned char", "unsigned short", "unsigned int",
                      "unsigned long", "unsigned long long"])]
mgeninteger = [Marray(["char", "signed char", "short", "int", "long", "long long",
                       "unsigned char", "unsigned short", "unsigned int",
                       "unsigned long", "unsigned long long"])]
sigeninteger = ["char", "signed char", "short", "int", "long", "long long"]
vigeninteger = [Vec(["int8_t", "int16_t", "int32_t", "int64_t", "long long"])] # long long non-standard. Deprecated.
migeninteger = [Marray(["char", "signed char", "short", "int", "long", "long long"])]
igeninteger = ["char", "signed char", "short", "int", "long", "long long",
                Vec(["int8_t", "int16_t", "int32_t", "int64_t", "long long"]), # long long non-standard. Deprecated.
                Marray(["char", "signed char", "short", "int", "long", "long long"])]
vugeninteger = [Vec(["uint8_t", "uint16_t", "uint32_t", "uint64_t",
                     "unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned long long"])] # Non-standard. Deprecated.
mugeninteger = [Marray(["unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned long long"])]
sugeninteger = ["unsigned char", "unsigned short", "unsigned int",
                "unsigned long", "unsigned long long"]
ugeninteger = [Vec(["uint8_t", "uint16_t", "uint32_t", "uint64_t",
                    "unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned long long"]), # Non-standard. Deprecated.
               Marray(["unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned long long"]),
               "unsigned char", "unsigned short", "unsigned int",
               "unsigned long", "unsigned long long"]
igenint32 = ["int32_t", Vec(["int32_t"]), Marray(["int32_t"])]
ugenint32 = ["uint32_t", Vec(["uint32_t"]), Marray(["uint32_t"])]
genint32 = ["int32_t", "uint32_t", Vec(["int32_t", "uint32_t"]), Marray(["int32_t", "uint32_t"])]

sgentype = ["char", "signed char", "short", "int", "long", "long long",
            "unsigned char", "unsigned short", "unsigned int",
            "unsigned long", "unsigned long long", "float", "double", "half"]
vgentype = [Vec(["int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t",
                 "uint32_t", "uint64_t", "float", "double", "half",
                 "long long", "unsigned long long"])] # long long non-standard. Deprecated.
mgentype = [Marray(["char", "signed char", "short", "int", "long", "long long",
                    "unsigned char", "unsigned short", "unsigned int",
                    "unsigned long", "unsigned long long", "float", "double", "half"])]

intptr = [MultiPtr("int"), RawPtr("int")]
floatptr = [MultiPtr("float"), RawPtr("float")]
doubleptr = [MultiPtr("double"), RawPtr("double")]
halfptr = [MultiPtr("half"), RawPtr("half")]
vfloatnptr = [MultiPtr(Vec(["float"])), RawPtr(Vec(["float"]))]
vdoublenptr = [MultiPtr(Vec(["double"])), RawPtr(Vec(["double"]))]
vhalfnptr = [MultiPtr(Vec(["half"])), RawPtr(Vec(["half"]))]
vgenfloatptr = [MultiPtr(Vec(["float", "double", "half"])),
                RawPtr(Vec(["float", "double", "half"]))]
mfloatnptr = [MultiPtr(Marray(["float"]))]
mdoublenptr = [MultiPtr(Marray(["double"]))]
mhalfnptr = [MultiPtr(Marray(["half"]))]
mgenfloatptr = [MultiPtr(Marray(["float", "double", "half"]))]
mintnptr = [MultiPtr(Marray(["int"]))]
vint32ptr = [MultiPtr(Vec(["int32_t"])), RawPtr(Vec(["int32_t"]))]

# To help resolve template arguments, these are given the index of their parent
# argument.
elementtype0 = [ElementType(0)]
unsignedtype0 = [ConversionTraitType("make_unsigned_t", 0)]
samesizesignedint0 = [ConversionTraitType("same_size_signed_int_t", 0)]
samesizeunsignedint0 = [ConversionTraitType("same_size_unsigned_int_t", 0)]
samesizefloat0 = [ConversionTraitType("same_size_float_t", 0)]
intelements0 = [ConversionTraitType("int_elements_t", 0)]
boolelements0 = [ConversionTraitType("bool_elements_t", 0)]
upsampledint0 = [ConversionTraitType("upsampled_int_t", 0)]

builtin_types = {
  "floatn" : floatn,
  "vfloatn" : vfloatn,
  "vfloat3or4" : vfloat3or4,
  "mfloatn" : mfloatn,
  "mfloat3or4" : mfloat3or4,
  "genfloatf" : genfloatf,
  "doublen" : doublen,
  "vdoublen" : vdoublen,
  "vdouble3or4" : vdouble3or4,
  "mdoublen" : mdoublen,
  "mdouble3or4" : mdouble3or4,
  "genfloatd" : genfloatd,
  "halfn" : halfn,
  "vhalfn" : vhalfn,
  "vhalf3or4" : vhalf3or4,
  "mhalfn" : mhalfn,
  "mhalf3or4" : mhalf3or4,
  "genfloath" : genfloath,
  "genfloat" : genfloat,
  "vgenfloat" : vgenfloat,
  "sgenfloat" : sgenfloat,
  "mgenfloat" : mgenfloat,
  "vgeofloat" : vgeofloat,
  "vgeodouble" : vgeodouble,
  "vgeohalf" : vgeohalf,
  "mgeofloat" : mgeofloat,
  "mgeodouble" : mgeodouble,
  "mgeohalf" : mgeohalf,
  "gengeofloat" : gengeofloat,
  "gengeodouble" : gengeodouble,
  "gengeohalf" : gengeohalf,
  "vint8n" : vint8n,
  "vint16n" : vint16n,
  "vint32n" : vint32n,
  "vint64n" : vint64n,
  "vint64n_ext" : vint64n_ext,
  "vuint8n" : vuint8n,
  "vuint16n" : vuint16n,
  "vuint32n" : vuint32n,
  "vuint64n" : vuint64n,
  "vuint64n_ext" : vuint64n_ext,
  "mint8n" : mint8n,
  "mint16n" : mint16n,
  "mint32n" : mint32n,
  "mint64n" : mint64n,
  "muint8n" : muint8n,
  "muint16n" : muint16n,
  "muint32n" : muint32n,
  "muint64n" : muint64n,
  "mcharn" : mcharn,
  "mshortn" : mshortn,
  "mintn" : mintn,
  "mushortn" : mushortn,
  "muintn" : muintn,
  "mulongn" : mulongn,
  "mbooln" : mbooln,
  "geninteger" : geninteger,
  "mgeninteger" : mgeninteger,
  "sigeninteger" : sigeninteger,
  "vigeninteger" : vigeninteger,
  "migeninteger" : migeninteger,
  "igeninteger" : igeninteger,
  "sugeninteger" : sugeninteger,
  "vugeninteger" : vugeninteger,
  "mugeninteger" : mugeninteger,
  "ugeninteger" : ugeninteger,
  "igenint32" : igenint32,
  "ugenint32" : ugenint32,
  "genint32" : genint32,
  "sgentype" : sgentype,
  "vgentype" : vgentype,
  "mgentype" : mgentype,
  "intptr" : intptr,
  "floatptr" : floatptr,
  "doubleptr" : doubleptr,
  "halfptr" : halfptr,
  "vfloatnptr" : vfloatnptr,
  "vdoublenptr" : vdoublenptr,
  "vhalfnptr" : vhalfnptr,
  "vgenfloatptr" : vgenfloatptr,
  "mfloatnptr" : mfloatnptr,
  "mdoublenptr" : mdoublenptr,
  "mhalfnptr" : mhalfnptr,
  "mgenfloatptr" : mgenfloatptr,
  "mintnptr" : mintnptr,
  "vint32nptr" : vint32ptr,
  "elementtype0" : elementtype0,
  "unsignedtype0" : unsignedtype0,
  "samesizesignedint0" : samesizesignedint0,
  "samesizeunsignedint0" : samesizeunsignedint0,
  "samesizefloat0" : samesizefloat0,
  "intelements0" : intelements0,
  "upsampledint0" : upsampledint0,
  "boolelements0" : boolelements0,
  "char" : ["char"],
  "signed char" : ["signed char"],
  "short" : ["short"],
  "int" : ["int"],
  "long" : ["long"],
  "long long" : ["long long"],
  "unsigned char" : ["unsigned char"],
  "unsigned short" : ["unsigned short"],
  "unsigned int" : ["unsigned int"],
  "unsigned long" : ["unsigned long"],
  "unsigned long long" : ["unsigned long long"],
  "float" : ["float"],
  "double" : ["double"],
  "half" : ["half"],
  "int8_t" : ["int8_t"],
  "int16_t" : ["int16_t"],
  "int32_t" : ["int32_t"],
  "int64_t" : ["int64_t"],
  "uint8_t" : ["uint8_t"],
  "uint16_t" : ["uint16_t"],
  "uint32_t" : ["uint32_t"],
  "uint64_t" : ["uint64_t"],
  "bool" : ["bool"]
}

### BUILTINS

def find_first_template_arg(arg_types):
  for arg_type in arg_types:
    if isinstance(arg_type, InstantiatedTemplatedArg):
      return arg_type
  return None

def is_marray_arg(arg_type):
  return isinstance(arg_type, InstantiatedTemplatedArg) and isinstance(arg_type.templated_type, Marray)

def is_vec_arg(arg_type):
  return isinstance(arg_type, InstantiatedTemplatedArg) and isinstance(arg_type.templated_type, Vec)

def convert_vec_arg_name(arg_type, arg_name):
  if is_vec_arg(arg_type):
    return f'typename detail::get_vec_t<{arg_type}>({arg_name})'
  return arg_name

class DefCommon:
  def __init__(self, return_type, arg_types, invoke_name, invoke_prefix,
               custom_invoke, size_alias, marray_use_loop):
    self.return_type = return_type
    self.arg_types = arg_types
    self.invoke_name = invoke_name
    self.invoke_prefix = invoke_prefix
    self.custom_invoke = custom_invoke
    self.size_alias = size_alias
    self.marray_use_loop = marray_use_loop

  def require_size_alias(self, alternative_name, marray_type):
    if not self.size_alias:
      # If there isn't a size alias defined, we add one.
      return (alternative_name, f'  constexpr size_t {alternative_name} = detail::num_elements<{marray_type.template_name}>::value;\n')
    return (self.size_alias, '')

  def convert_loop_arg(self, arg_type, arg_name):
    if isinstance(arg_type, MultiPtr):
      return f'address_space_cast<Space, IsDecorated, detail::get_elem_type_t<{arg_type.element_type}>>(&(*{arg_name})[I])'
    if is_marray_arg(arg_type):
      return str(arg_name) + '[I]'
    return str(arg_name)

  def get_marray_loop_invoke_body(self, namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type):
    result = ""
    args = [self.convert_loop_arg(arg_type, arg_name) for arg_type, arg_name in zip(arg_types, arg_names)]
    joined_args = ', '.join(args)
    (size_alias, size_alias_init) = self.require_size_alias('N', first_marray_type)
    result = result + size_alias_init
    return result + f"""  {return_type} Res;
  for (int I = 0; I < {size_alias}; ++I)
    Res[I] = {namespaced_builtin_name}({joined_args});
  return Res;"""

  def get_marray_vec_cast_invoke_body(self, namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type):
    result = ""
    vec_cast_args = [f'detail::to_vec({arg_name})' if is_marray_arg(arg_type) else str(arg_name) for arg_type, arg_name in zip(arg_types, arg_names)]
    joined_vec_cast_args = ', '.join(vec_cast_args)
    (size_alias, size_alias_init) = self.require_size_alias('N', first_marray_type)
    result = result + size_alias_init
    vec_call = f'{namespaced_builtin_name}({joined_vec_cast_args})'
    if isinstance(return_type, InstantiatedTemplatedReturnType):
      # Convert the vec call result to marray.
      vec_call = f'detail::to_marray({vec_call})'
    return result + f'  return {vec_call};'

  def get_marray_vectorized_invoke_body(self, namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type):
    result = ""
    (size_alias, size_alias_init) = self.require_size_alias('N', first_marray_type)
    result = result + size_alias_init
    # Adjust arguments for partial results and the remaining work at the end.
    imm_args = []
    rem_args = []
    for arg_type, arg_name in zip(arg_types, arg_names):
      is_marray = is_marray_arg(arg_type)
      imm_args.append(f'detail::to_vec2({arg_name}, I * 2)' if is_marray else arg_name)
      rem_args.append(f'{arg_name}[{size_alias} - 1]' if is_marray else arg_name)
    joined_imm_args = ', '.join(imm_args)
    joined_rem_args = ', '.join(rem_args)
    return result + f"""  {return_type} Res;
  for (size_t I = 0; I < {size_alias} / 2; ++I) {{
    auto PartialRes = {namespaced_builtin_name}({joined_imm_args});
    std::memcpy(&Res[I * 2], &PartialRes, sizeof(decltype(PartialRes)));
  }}
  if ({size_alias} % 2)
    Res[{size_alias} - 1] = {namespaced_builtin_name}({joined_rem_args});
  return Res;"""

  def get_marray_invoke_body(self, namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type):
    # If the associated marray types have restriction on their sizes, we assume
    # they can be converted directly to vector.
    if first_marray_type.templated_type.valid_sizes:
      return self.get_marray_vec_cast_invoke_body(namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type)
    # If there is a pointer argument, we need to use the simple loop solution.
    if self.marray_use_loop or any([isinstance(arg_type, MultiPtr) for arg_type in arg_types]):
      return self.get_marray_loop_invoke_body(namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type)
    # Otherwise, we vectorize the body.
    return self.get_marray_vectorized_invoke_body(namespaced_builtin_name, return_type, arg_types, arg_names, first_marray_type)

  def get_invoke_body(self, builtin_name, namespace, invoke_name, return_type, arg_types, arg_names):
    for arg_type in arg_types:
      if is_marray_arg(arg_type):
        namespaced_builtin_name = f'{namespace}::{builtin_name}' if namespace else builtin_name
        return self.get_marray_invoke_body(namespaced_builtin_name, return_type, arg_types, arg_names, arg_type)
    return self.get_scalar_vec_invoke_body(invoke_name, return_type, arg_types, arg_names)

  def get_invoke(self, builtin_name, namespace, return_type, arg_types, arg_names):
    if self.custom_invoke:
      return self.custom_invoke(return_type, arg_types, arg_names)
    invoke_name = self.invoke_name if self.invoke_name else builtin_name
    result = ""
    if self.size_alias:
      template_arg = find_first_template_arg(arg_types)
      if template_arg:
        result = result + f'  constexpr size_t {self.size_alias} = detail::num_elements<{template_arg.template_name}>::value;'
    return result + self.get_invoke_body(builtin_name, namespace, invoke_name, return_type, arg_types, arg_names)


class Def(DefCommon):
  def __init__(self, return_type, arg_types, invoke_name=None,
               invoke_prefix="", custom_invoke=None, fast_math_invoke_name=None,
               convert_args=[], size_alias=None, marray_use_loop=False):
    super().__init__(return_type, arg_types, invoke_name, invoke_prefix,
                     custom_invoke, size_alias, marray_use_loop)
    self.fast_math_invoke_name = fast_math_invoke_name
    # List of tuples with mappings for arguments to cast to argument types.
    # First element in a tuple is the index of the argument to cast and the
    # second element is the index of the argument type to convert to.
    # Alternatively, the second element can be a string representation of the
    # conversion function or type.
    self.convert_args = convert_args

  def get_invoke_args(self, arg_types, arg_names):
    result = list(map(convert_vec_arg_name, arg_types, arg_names))
    for (arg_idx, type_conv) in self.convert_args:
      # type_conv is either an index or a conversion function/type.
      conv = type_conv if isinstance(type_conv, str) else arg_types[type_conv]
      result[arg_idx] = f'{conv}({result[arg_idx]})'
    return result

  def get_scalar_vec_invoke_body(self, invoke_name, return_type, arg_types, arg_names):
    invoke_args = self.get_invoke_args(arg_types, arg_names)
    result = ""
    if self.fast_math_invoke_name:
      result = result + f"""  if constexpr (detail::use_fast_math_v<{arg_types[0]}>) {{
    return __sycl_std::__invoke_{self.fast_math_invoke_name}<{return_type}>({(", ".join(invoke_args))});
  }}\n"""
    return result + f'  return __sycl_std::__invoke_{self.invoke_prefix}{invoke_name}<{return_type}>({(", ".join(invoke_args))});'

class RelDef(DefCommon):
  def __init__(self, return_type, arg_types, invoke_name=None,
               invoke_prefix="", custom_invoke=None):
    # NOTE: Relational builtins never use the vectorized solution as the vectors
    #       are likely to use values larger than bool.
    super().__init__(return_type, arg_types, invoke_name, invoke_prefix,
                     custom_invoke, None, True)

  def get_scalar_vec_invoke_body(self, invoke_name, return_type, arg_types, arg_names):
    if self.custom_invoke:
      return self.custom_invoke(return_type, arg_types, arg_names)
    invoke_args = ', '.join(arg_names)
    return f'  return detail::RelConverter<{return_type}>::apply(__sycl_std::__invoke_{self.invoke_prefix}{invoke_name}<detail::internal_rel_ret_t<{return_type}>>({invoke_args}));'

def custom_signed_abs_scalar_invoke(return_type, _, arg_names):
  args = ' ,'.join(arg_names)
  return f'return static_cast<{return_type}>(__sycl_std::__invoke_s_abs<detail::make_unsigned_t<{return_type}>>({args}));'

def custom_signed_abs_vector_invoke(return_type, _, arg_names):
  args = ' ,'.join(arg_names)
  return f'return __sycl_std::__invoke_s_abs<detail::make_unsigned_t<{return_type}>>({args}).template convert<detail::get_elem_type_t<{return_type}>>();'

def get_custom_any_all_vec_invoke(invoke_name):
  return (lambda _, arg_types, arg_names: f"""  return detail::rel_sign_bit_test_ret_t<{arg_types[0]}>(
      __sycl_std::__invoke_{invoke_name}<detail::rel_sign_bit_test_ret_t<{arg_types[0]}>>(
          detail::rel_sign_bit_test_arg_t<{arg_types[0]}>({arg_names[0]})));""")

def custom_bool_select_invoke(return_type, _, arg_names):
  return f"""  return __sycl_std::__invoke_select<{return_type}>(
      {arg_names[0]}, {arg_names[1]}, static_cast<detail::get_select_opencl_builtin_c_arg_type<{return_type}>>({arg_names[2]}));"""

def get_custom_any_all_marray_invoke(builtin):
  return (lambda _, arg_types, arg_names: f'  return std::{builtin}_of({arg_names[0]}.begin(), {arg_names[0]}.end(), [](detail::get_elem_type_t<{arg_types[0]}> X) {{ return {builtin}(X); }});')

sycl_builtins = {# Math functions
                 "acos": [Def("genfloat", ["genfloat"])],
                 "acosh": [Def("genfloat", ["genfloat"])],
                 "acospi": [Def("genfloat", ["genfloat"])],
                 "asin": [Def("genfloat", ["genfloat"])],
                 "asinh": [Def("genfloat", ["genfloat"])],
                 "asinpi": [Def("genfloat", ["genfloat"])],
                 "atan": [Def("genfloat", ["genfloat"])],
                 "atan2": [Def("genfloat", ["genfloat", "genfloat"])],
                 "atanh": [Def("genfloat", ["genfloat"])],
                 "atanpi": [Def("genfloat", ["genfloat"])],
                 "atan2pi": [Def("genfloat", ["genfloat", "genfloat"])],
                 "cbrt": [Def("genfloat", ["genfloat"])],
                 "ceil": [Def("genfloat", ["genfloat"])],
                 "copysign": [Def("genfloat", ["genfloat", "genfloat"])],
                 "cos": [Def("genfloat", ["genfloat"], fast_math_invoke_name = "native_cos")],
                 "cosh": [Def("genfloat", ["genfloat"])],
                 "cospi": [Def("genfloat", ["genfloat"])],
                 "erfc": [Def("genfloat", ["genfloat"])],
                 "erf": [Def("genfloat", ["genfloat"])],
                 "exp": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp")],
                 "exp2": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp2")],
                 "exp10": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp10")],
                 "expm1": [Def("genfloat", ["genfloat"])],
                 "fabs": [Def("genfloat", ["genfloat"])],
                 "fdim": [Def("genfloat", ["genfloat", "genfloat"])],
                 "floor": [Def("genfloat", ["genfloat"])],
                 "fma": [Def("genfloat", ["genfloat", "genfloat", "genfloat"])],
                 "fmax": [Def("genfloat", ["genfloat", "genfloat"]),
                          Def("vgenfloat", ["vgenfloat", "elementtype0"], convert_args=[(1,0)]),
                          Def("mgenfloat", ["mgenfloat", "elementtype0"])],
                 "fmin": [Def("genfloat", ["genfloat", "genfloat"]),
                          Def("vgenfloat", ["vgenfloat", "elementtype0"], convert_args=[(1,0)]),
                          Def("mgenfloat", ["mgenfloat", "elementtype0"])],
                 "fmod": [Def("genfloat", ["genfloat", "genfloat"])],
                 "fract": [Def("vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                           Def("mgenfloat", ["mgenfloat", "mgenfloatptr"]),
                           Def("float", ["float", "floatptr"]),
                           Def("double", ["double", "doubleptr"]),
                           Def("half", ["half", "halfptr"])],
                 "frexp": [Def("vgenfloat", ["vgenfloat", "vint32nptr"]),
                           Def("mgenfloat", ["mgenfloat", "mintnptr"], marray_use_loop=True),
                           Def("float", ["float", "intptr"]),
                           Def("double", ["double", "intptr"]),
                           Def("half", ["half", "intptr"])],
                 "hypot": [Def("genfloat", ["genfloat", "genfloat"])],
                 "ilogb": [Def("intelements0", ["vgenfloat"]),
                           Def("intelements0", ["mgenfloat"]),
                           Def("int", ["float"]),
                           Def("int", ["double"]),
                           Def("int", ["half"])],
                 "ldexp": [Def("vgenfloat", ["vgenfloat", "vint32n"]),
                           Def("vgenfloat", ["vgenfloat", "int"], convert_args=[(1,"vec<int, N>")], size_alias="N"),
                           Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
                           Def("mgenfloat", ["mgenfloat", "int"]),
                           Def("float", ["float", "int"]),
                           Def("double", ["double", "int"]),
                           Def("half", ["half", "int"])],
                 "lgamma": [Def("genfloat", ["genfloat"])],
                 "lgamma_r": [Def("vgenfloat", ["vgenfloat", "vint32nptr"]),
                              Def("mgenfloat", ["mgenfloat", "mintnptr"]),
                              Def("float", ["float", "intptr"]),
                              Def("double", ["double", "intptr"]),
                              Def("half", ["half", "intptr"])],
                 "log": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log")],
                 "log2": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log2")],
                 "log10": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log10")],
                 "log1p": [Def("genfloat", ["genfloat"])],
                 "logb": [Def("genfloat", ["genfloat"])],
                 "mad": [Def("genfloat", ["genfloat", "genfloat", "genfloat"])],
                 "maxmag": [Def("genfloat", ["genfloat", "genfloat"])],
                 "minmag": [Def("genfloat", ["genfloat", "genfloat"])],
                 "modf": [Def("vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                          Def("mgenfloat", ["mgenfloat", "mgenfloatptr"]),
                          Def("float", ["float", "floatptr"]),
                          Def("double", ["double", "doubleptr"]),
                          Def("half", ["half", "halfptr"])],
                 "nan": [Def("samesizefloat0", ["vuint32n"]),
                         Def("samesizefloat0", ["muintn"], marray_use_loop=True),
                         Def("samesizefloat0", ["unsigned int"]),
                         Def("samesizefloat0", ["vuint64n_ext"]),
                         Def("samesizefloat0", ["mulongn"], marray_use_loop=True),
                         Def("samesizefloat0", ["unsigned long"]),
                         Def("samesizefloat0", ["unsigned long long"]),
                         Def("samesizefloat0", ["vuint16n"]),
                         Def("samesizefloat0", ["mushortn"], marray_use_loop=True),
                         Def("samesizefloat0", ["unsigned short"])],
                 "nextafter": [Def("genfloat", ["genfloat", "genfloat"])],
                 "pow": [Def("genfloat", ["genfloat", "genfloat"])],
                 "pown": [Def("vgenfloat", ["vgenfloat", "vint32n"]),
                          Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
                          Def("float", ["float", "int"]),
                          Def("double", ["double", "int"]),
                          Def("half", ["half", "int"])],
                 "powr": [Def("genfloat", ["genfloat", "genfloat"], fast_math_invoke_name="native_powr")],
                 "remainder": [Def("genfloat", ["genfloat", "genfloat"])],
                 "remquo": [Def("vgenfloat", ["vgenfloat", "vgenfloat", "vint32nptr"]),
                            Def("mgenfloat", ["mgenfloat", "mgenfloat", "mintnptr"], marray_use_loop=True),
                            Def("float", ["float", "float", "intptr"]),
                            Def("double", ["double", "double", "intptr"]),
                            Def("half", ["half", "half", "intptr"])],
                 "rint": [Def("genfloat", ["genfloat"])],
                 "rootn": [Def("vgenfloat", ["vgenfloat", "vint32n"]),
                           Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
                           Def("float", ["float", "int"]),
                           Def("double", ["double", "int"]),
                           Def("half", ["half", "int"])],
                 "round": [Def("genfloat", ["genfloat"])],
                 "rsqrt": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_rsqrt")],
                 "sin": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_sin")],
                 "sincos": [Def("vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                            Def("mgenfloat", ["mgenfloat", "mgenfloatptr"]),
                            Def("float", ["float", "floatptr"]),
                            Def("double", ["double", "doubleptr"]),
                            Def("half", ["half", "halfptr"])],
                 "sinh": [Def("genfloat", ["genfloat"])],
                 "sinpi": [Def("genfloat", ["genfloat"])],
                 "sqrt": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_sqrt")],
                 "tan": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_tan")],
                 "tanh": [Def("genfloat", ["genfloat"])],
                 "tanpi": [Def("genfloat", ["genfloat"])],
                 "tgamma": [Def("genfloat", ["genfloat"])],
                 "trunc": [Def("genfloat", ["genfloat"])],
                 # Integer functions
                 "abs_diff": [Def("unsignedtype0", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                              Def("unsignedtype0", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "add_sat": [Def("igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                             Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "hadd": [Def("igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                          Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "rhadd": [Def("igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                           Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "clz": [Def("geninteger", ["geninteger"], marray_use_loop=True)],
                 "ctz": [Def("geninteger", ["geninteger"], marray_use_loop=True)],
                 "mad_hi": [Def("igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                            Def("ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "mad_sat": [Def("igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                             Def("ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "mul_hi": [Def("igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                            Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "rotate": [Def("geninteger", ["geninteger", "geninteger"], marray_use_loop=True)],
                 "sub_sat": [Def("igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                             Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 "upsample": [Def("upsampledint0", ["int8_t", "uint8_t"], invoke_prefix="s_"),
                              Def("upsampledint0", ["char", "uint8_t"], invoke_prefix="s_"), # TODO: Non-standard. Deprecate.
                              Def("upsampledint0", ["vint8n", "vuint8n"], invoke_prefix="s_"),
                              Def("upsampledint0", ["mint8n", "muint8n"]),
                              Def("upsampledint0", ["uint8_t", "uint8_t"], invoke_prefix="u_"),
                              Def("upsampledint0", ["vuint8n", "vuint8n"], invoke_prefix="u_"),
                              Def("upsampledint0", ["muint8n", "muint8n"]),
                              Def("upsampledint0", ["int16_t", "uint16_t"], invoke_prefix="s_"),
                              Def("upsampledint0", ["vint16n", "vuint16n"], invoke_prefix="s_"),
                              Def("upsampledint0", ["mint16n", "muint16n"]),
                              Def("upsampledint0", ["uint16_t", "uint16_t"], invoke_prefix="u_"),
                              Def("upsampledint0", ["vuint16n", "vuint16n"], invoke_prefix="u_"),
                              Def("upsampledint0", ["muint16n", "muint16n"]),
                              Def("upsampledint0", ["int32_t", "uint32_t"], invoke_prefix="s_"),
                              Def("upsampledint0", ["vint32n", "vuint32n"], invoke_prefix="s_"),
                              Def("upsampledint0", ["mint32n", "muint32n"]),
                              Def("upsampledint0", ["uint32_t", "uint32_t"], invoke_prefix="u_"),
                              Def("upsampledint0", ["vuint32n", "vuint32n"], invoke_prefix="u_"),
                              Def("upsampledint0", ["muint32n", "muint32n"])],
                 "popcount": [Def("geninteger", ["geninteger"], marray_use_loop=True)],
                 "mad24": [Def("igenint32", ["igenint32", "igenint32", "igenint32"], invoke_prefix="s_"),
                           Def("ugenint32", ["ugenint32", "ugenint32", "ugenint32"], invoke_prefix="u_")],
                 "mul24": [Def("igenint32", ["igenint32", "igenint32"], invoke_prefix="s_"),
                           Def("ugenint32", ["ugenint32", "ugenint32"], invoke_prefix="u_")],
                 # Common functions
                 "clamp": [Def("genfloat", ["genfloat", "genfloat", "genfloat"], invoke_prefix="f"),
                           Def("vfloatn", ["vfloatn", "float", "float"], invoke_prefix="f", convert_args=[(1,0),(2,0)]),
                           Def("vdoublen", ["vdoublen", "double", "double"], invoke_prefix="f", convert_args=[(1,0),(2,0)]),
                           Def("vhalfn", ["vhalfn", "half", "half"], invoke_prefix="f", convert_args=[(1,0),(2,0)]), # Non-standard. Deprecated.
                           Def("igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_", marray_use_loop=True),
                           Def("ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_", marray_use_loop=True),
                           Def("vigeninteger", ["vigeninteger", "elementtype0", "elementtype0"], invoke_prefix="s_"),
                           Def("vugeninteger", ["vugeninteger", "elementtype0", "elementtype0"], invoke_prefix="u_"),
                           Def("mgentype", ["mgentype", "elementtype0", "elementtype0"], marray_use_loop=True)],
                 "degrees": [Def("genfloat", ["genfloat"])],
                 "(max)": [Def("genfloat", ["genfloat", "genfloat"], invoke_name="fmax_common"),
                           Def("vfloatn", ["vfloatn", "float"], invoke_name="fmax_common", convert_args=[(1,0)]),
                           Def("vdoublen", ["vdoublen", "double"], invoke_name="fmax_common", convert_args=[(1,0)]),
                           Def("vhalfn", ["vhalfn", "half"], invoke_name="fmax_common", convert_args=[(1,0)]), # Non-standard. Deprecated.
                           Def("igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_max", marray_use_loop=True),
                           Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_max", marray_use_loop=True),
                           Def("vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_max"),
                           Def("vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_max"),
                           Def("mgentype", ["mgentype", "elementtype0"], marray_use_loop=True)],
                 "(min)": [Def("genfloat", ["genfloat", "genfloat"], invoke_name="fmin_common"),
                           Def("vfloatn", ["vfloatn", "float"], invoke_name="fmin_common", convert_args=[(1,0)]),
                           Def("vdoublen", ["vdoublen", "double"], invoke_name="fmin_common", convert_args=[(1,0)]),
                           Def("vhalfn", ["vhalfn", "half"], invoke_name="fmax_common", convert_args=[(1,0)]), # Non-standard. Deprecated.
                           Def("igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_min", marray_use_loop=True),
                           Def("ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_min", marray_use_loop=True),
                           Def("vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_min"),
                           Def("vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_min"),
                           Def("mgentype", ["mgentype", "elementtype0"], marray_use_loop=True)],
                 "mix": [Def("genfloat", ["genfloat", "genfloat", "genfloat"]),
                         Def("vfloatn", ["vfloatn", "vfloatn", "float"], convert_args=[(2,0)]),
                         Def("vdoublen", ["vdoublen", "vdoublen", "double"], convert_args=[(2,0)]),
                         Def("vhalfn", ["vhalfn", "vhalfn", "half"], convert_args=[(2,0)]), # Non-standard. Deprecated.
                         Def("mfloatn", ["mfloatn", "mfloatn", "float"]),
                         Def("mdoublen", ["mdoublen", "mdoublen", "double"]),
                         Def("mhalfn", ["mhalfn", "mhalfn", "half"])], # Non-standard. Deprecated.
                 "radians": [Def("genfloat", ["genfloat"])],
                 "step": [Def("genfloat", ["genfloat", "genfloat"]),
                          Def("vfloatn", ["float", "vfloatn"], convert_args=[(0,1)]),
                          Def("vdoublen", ["double", "vdoublen"], convert_args=[(0,1)]),
                          Def("vhalfn", ["half", "vhalfn"], convert_args=[(0,1)]), # Non-standard. Deprecated.
                          Def("mfloatn", ["float", "mfloatn"]),
                          Def("mdoublen", ["double", "mdoublen"]),
                          Def("mhalfn", ["half", "mhalfn"])], # Non-standard. Deprecated.
                 "smoothstep": [Def("genfloat", ["genfloat", "genfloat", "genfloat"]),
                                Def("vfloatn", ["float", "float", "vfloatn"], convert_args=[(0,2),(1,2)]),
                                Def("vdoublen", ["double", "double", "vdoublen"], convert_args=[(0,2),(1,2)]),
                                Def("vhalfn", ["half", "half", "vhalfn"], convert_args=[(0,2),(1,2)]), # Non-standard. Deprecated.
                                Def("mfloatn", ["float", "float", "mfloatn"]),
                                Def("mdoublen", ["double", "double", "mdoublen"]),
                                Def("mhalfn", ["half", "half", "mhalfn"])],
                 "sign": [Def("genfloat", ["genfloat"])],
                 "abs": [Def("genfloat", ["genfloat"], invoke_prefix="f"), # TODO: Non-standard. Deprecate.
                         Def("sigeninteger", ["sigeninteger"], custom_invoke=custom_signed_abs_scalar_invoke),
                         Def("vigeninteger", ["vigeninteger"], custom_invoke=custom_signed_abs_vector_invoke),
                         Def("migeninteger", ["migeninteger"], marray_use_loop=True),
                         Def("ugeninteger", ["ugeninteger"], invoke_prefix="u_", marray_use_loop=True)],
                 # Geometric functions
                 "cross": [Def("vfloat3or4", ["vfloat3or4", "vfloat3or4"]),
                           Def("vdouble3or4", ["vdouble3or4", "vdouble3or4"]),
                           Def("vhalf3or4", ["vhalf3or4", "vhalf3or4"]), # TODO: Non-standard. Deprecate.
                           Def("mfloat3or4", ["mfloat3or4", "mfloat3or4"]),
                           Def("mdouble3or4", ["mdouble3or4", "mdouble3or4"]),
                           Def("mhalf3or4", ["mhalf3or4", "mhalf3or4"])], # TODO: Non-standard. Deprecate.
                 "dot": [Def("float", ["vgeofloat", "vgeofloat"], invoke_name="Dot"),
                         Def("double", ["vgeodouble", "vgeodouble"], invoke_name="Dot"),
                         Def("half", ["vgeohalf", "vgeohalf"], invoke_name="Dot"), # TODO: Non-standard. Deprecate.
                         Def("float", ["mgeofloat", "mgeofloat"], invoke_name="Dot"),
                         Def("double", ["mgeodouble", "mgeodouble"], invoke_name="Dot"),
                         Def("half", ["mgeohalf", "mgeohalf"], invoke_name="Dot"), # TODO: Non-standard. Deprecate.
                         Def("sgenfloat", ["sgenfloat", "sgenfloat"], custom_invoke=(lambda return_types, arg_types, arg_names: '  return ' + ' * '.join(arg_names) + ';'))],
                 "distance": [Def("float", ["gengeofloat", "gengeofloat"]),
                              Def("double", ["gengeodouble", "gengeodouble"]),
                              Def("half", ["gengeohalf", "gengeohalf"])], # TODO: Non-standard. Deprecate.
                 "length": [Def("float", ["gengeofloat"]),
                            Def("double", ["gengeodouble"]),
                            Def("half", ["gengeohalf"])], # TODO: Non-standard. Deprecate.
                 "normalize": [Def("gengeofloat", ["gengeofloat"]),
                               Def("gengeodouble", ["gengeodouble"]),
                               Def("gengeohalf", ["gengeohalf"])], # TODO: Non-standard. Deprecate.
                 "fast_distance": [Def("float", ["gengeofloat", "gengeofloat"]),
                                   Def("double", ["gengeodouble", "gengeodouble"])],
                 "fast_length": [Def("float", ["gengeofloat"]),
                                 Def("double", ["gengeodouble"])],
                 "fast_normalize": [Def("gengeofloat", ["gengeofloat"]),
                                    Def("gengeodouble", ["gengeodouble"])],
                 # Relational functions
                 "isequal": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdEqual"),
                             RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdEqual"),
                             RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isnotequal": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FUnordNotEqual"),
                                RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FUnordNotEqual"),
                                RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isgreater": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdGreaterThan"),
                               RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThan"),
                               RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isgreaterequal": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdGreaterThanEqual"),
                                    RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThanEqual"),
                                    RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isless": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdLessThan"),
                            RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThan"),
                            RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "islessequal": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdLessThanEqual"),
                                 RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThanEqual"),
                                 RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "islessgreater": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdNotEqual"),
                                   RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdNotEqual"),
                                   RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isfinite": [RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsFinite"),
                              RelDef("bool", ["sgenfloat"], invoke_name="IsFinite"),
                              RelDef("boolelements0", ["mgenfloat"])],
                 "isinf": [RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsInf"),
                           RelDef("bool", ["sgenfloat"], invoke_name="IsInf"),
                           RelDef("boolelements0", ["mgenfloat"])],
                 "isnan": [RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsNan"),
                           RelDef("bool", ["sgenfloat"], invoke_name="IsNan"),
                           RelDef("boolelements0", ["mgenfloat"])],
                 "isnormal": [RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsNormal"),
                              RelDef("bool", ["sgenfloat"], invoke_name="IsNormal"),
                              RelDef("boolelements0", ["mgenfloat"])],
                 "isordered": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Ordered"),
                               RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="Ordered"),
                               RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "isunordered": [RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Unordered"),
                                 RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="Unordered"),
                                 RelDef("boolelements0", ["mgenfloat", "mgenfloat"])],
                 "signbit": [RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="SignBitSet"),
                             RelDef("bool", ["sgenfloat"], invoke_name="SignBitSet"),
                             RelDef("boolelements0", ["mgenfloat"])],
                 "any": [Def("int", ["vigeninteger"], custom_invoke=get_custom_any_all_vec_invoke("Any")),
                         Def("bool", ["sigeninteger"], custom_invoke=(lambda return_type, arg_types, arg_names: f'  return detail::Boolean<1>(int(detail::msbIsSet({arg_names[0]})));')),
                         Def("bool", ["migeninteger"], custom_invoke=get_custom_any_all_marray_invoke("any"))],
                 "all": [Def("int", ["vigeninteger"], custom_invoke=get_custom_any_all_vec_invoke("All")),
                         Def("bool", ["sigeninteger"], custom_invoke=(lambda return_type, arg_types, arg_names: f'  return detail::Boolean<1>(int(detail::msbIsSet({arg_names[0]})));')),
                         Def("bool", ["migeninteger"], custom_invoke=get_custom_any_all_marray_invoke("all"))],
                 "bitselect": [Def("vgentype", ["vgentype", "vgentype", "vgentype"]),
                               Def("sgentype", ["sgentype", "sgentype", "sgentype"]),
                               Def("mgentype", ["mgentype", "mgentype", "mgentype"], marray_use_loop=True)],
                 "select": [Def("vint8n", ["vint8n", "vint8n", "vint8n"]),
                            Def("vint16n", ["vint16n", "vint16n", "vint16n"]),
                            Def("vint32n", ["vint32n", "vint32n", "vint32n"]),
                            Def("vint64n_ext", ["vint64n_ext", "vint64n_ext", "vint64n_ext"]),
                            Def("vuint8n", ["vuint8n", "vuint8n", "vint8n"]),
                            Def("vuint16n", ["vuint16n", "vuint16n", "vint16n"]),
                            Def("vuint32n", ["vuint32n", "vuint32n", "vint32n"]),
                            Def("vuint64n_ext", ["vuint64n_ext", "vuint64n_ext", "vint64n_ext"]),
                            Def("vfloatn", ["vfloatn", "vfloatn", "vint32n"]),
                            Def("vdoublen", ["vdoublen", "vdoublen", "vint64n_ext"]),
                            Def("vhalfn", ["vhalfn", "vhalfn", "vint16n"]),
                            Def("vint8n", ["vint8n", "vint8n", "vuint8n"]),
                            Def("vint16n", ["vint16n", "vint16n", "vuint16n"]),
                            Def("vint32n", ["vint32n", "vint32n", "vuint32n"]),
                            Def("vint64n_ext", ["vint64n_ext", "vint64n_ext", "vuint64n_ext"]),
                            Def("vuint8n", ["vuint8n", "vuint8n", "vuint8n"]),
                            Def("vuint16n", ["vuint16n", "vuint16n", "vuint16n"]),
                            Def("vuint32n", ["vuint32n", "vuint32n", "vuint32n"]),
                            Def("vuint64n_ext", ["vuint64n_ext", "vuint64n_ext", "vuint64n_ext"]),
                            Def("vfloatn", ["vfloatn", "vfloatn", "vuint32n"]),
                            Def("vdoublen", ["vdoublen", "vdoublen", "vuint64n_ext"]),
                            Def("vhalfn", ["vhalfn", "vhalfn", "vuint16n"]),
                            Def("sgentype", ["sgentype", "sgentype", "bool"], custom_invoke=custom_bool_select_invoke),
                            Def("mgentype", ["mgentype", "mgentype", "mbooln"], marray_use_loop=True)]}
native_builtins = {"cos": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "divide": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_")],
                   "exp": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "exp2": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "exp10": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "log": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "log2": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "log10": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "powr": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_")],
                   "recip": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "rsqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "sin": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "sqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
                   "tan": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")]}
half_precision_builtins = {"cos": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "divide": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_")],
                           "exp": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "exp2": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "exp10": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "log": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "log2": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "log10": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "powr": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_")],
                           "recip": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "rsqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "sin": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "sqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
                           "tan": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")]}

builtins_groups = [(None, sycl_builtins),
                   ("native", native_builtins),
                   ("half_precision", half_precision_builtins)]

### GENERATION

def select_from_mapping(mappings, arg_types, arg_type):
  mapping = mappings[arg_type]
  # In some cases we may need to limit definitions to smaller than geninteger so
  # check for the different possible ones.
  if isinstance(mapping, ElementType):
    parent_mapping = mappings[arg_types[mapping.parent_idx]]
    return InstantiatedElementType(parent_mapping, mapping.parent_idx)
  if isinstance(mapping, ConversionTraitType):
    parent_mapping = mappings[arg_types[mapping.parent_idx]]
    return InstantiatedConversionTraitType(parent_mapping, mapping.trait, mapping.parent_idx)
  return mapping

def instantiate_arg(idx, arg):
  if isinstance(arg, TemplatedType):
    return InstantiatedTemplatedArg(f'T{idx}', arg)
  if isinstance(arg, MultiPtr):
    return MultiPtr(instantiate_arg(idx, arg.element_type))
  if isinstance(arg, RawPtr):
    return RawPtr(instantiate_arg(idx, arg.element_type))
  if isinstance(arg, InstantiatedElementType):
    return InstantiatedElementType(instantiate_arg(arg.parent_idx, arg.referenced_type), arg.parent_idx)
  if isinstance(arg, InstantiatedConversionTraitType):
    return InstantiatedConversionTraitType(instantiate_arg(arg.parent_idx, arg.parent_type), arg.trait, arg.parent_idx)
  return arg

def instantiate_return_type(return_type, instantiated_args):
  if isinstance(return_type, TemplatedType):
    first_templated = find_first_template_arg(instantiated_args)
    return InstantiatedTemplatedReturnType(str(first_templated), return_type)
  if isinstance(return_type, MultiPtr):
    return MultiPtr(instantiate_return_type(return_type.element_type, instantiated_args))
  if isinstance(return_type, RawPtr):
    return RawPtr(instantiate_return_type(return_type.element_type, instantiated_args))
  if isinstance(return_type, InstantiatedElementType):
    return InstantiatedElementType(instantiate_return_type(return_type.referenced_type, instantiated_args), return_type.parent_idx)
  if isinstance(return_type, InstantiatedConversionTraitType):
    return InstantiatedConversionTraitType(instantiate_return_type(return_type.parent_type, instantiated_args), return_type.trait, return_type.parent_idx)
  return return_type

def is_valid_combination(return_type, arg_types):
  marray_arg_seen = False
  vec_arg_seen = False
  raw_ptr_seen = False
  for t in [return_type] + arg_types:
    if is_marray_arg(t):
      marray_arg_seen = True
    elif is_vec_arg(t):
      vec_arg_seen = True
    elif isinstance(t, RawPtr):
      raw_ptr_seen = True
  # Following rules apply:
  # 1. vec and marray cannot be in the same combination.
  # 2. marray and raw pointers were never supported together and shouldn't be.
  return not (marray_arg_seen and (vec_arg_seen or raw_ptr_seen))

def type_combinations(return_type, arg_types):
  unique_types = list(dict.fromkeys(arg_types + [return_type]))
  unique_type_lists = [builtin_types[unique_type] for unique_type in unique_types]
  combinations = list(itertools.product(*unique_type_lists))
  result = []
  for combination in combinations:
    mappings = dict(zip(unique_types, combination))
    mapped_return_type = select_from_mapping(mappings, arg_types, return_type)
    mapped_arg_types = [select_from_mapping(mappings, arg_types, arg_type) for arg_type in arg_types]
    instantiated_arg_types = [instantiate_arg(idx, arg_type) for idx, arg_type in enumerate(mapped_arg_types)]
    instantiated_return_type = instantiate_return_type(mapped_return_type, instantiated_arg_types)
    if is_valid_combination(instantiated_return_type, instantiated_arg_types):
      result.append((instantiated_return_type, instantiated_arg_types))
  return result

def get_all_template_args(arg_types):
  result = []
  for arg_type in arg_types:
    if isinstance(arg_type, InstantiatedTemplatedArg):
      result.append(arg_type)
    if isinstance(arg_type, MultiPtr) or isinstance(arg_type, RawPtr):
      result = result + get_all_template_args([arg_type.element_type])
  return result

def get_vec_arg_requirement(vec_arg):
  return '(' + (' && '.join(vec_arg.get_requirements())) + ')'

def get_func_return(return_type, arg_types):
  temp_args = get_all_template_args(arg_types)
  if len(temp_args) > 0:
    conjunc_reqs = ' && '.join([get_vec_arg_requirement(temp_arg) for temp_arg in temp_args])
    return f'std::enable_if_t<{conjunc_reqs}, {return_type}>'
  return str(return_type)

def get_template_args(return_type, arg_types):
  temp_args = get_all_template_args(arg_types)
  result = [f'typename {temp_arg.template_name}' for temp_arg in temp_args]
  for t in ([return_type] + arg_types):
    if isinstance(t, MultiPtr):
      result.append('access::address_space Space')
      result.append('access::decorated IsDecorated')
  return result

def get_deprecation(builtin, return_type, arg_types):
  # TODO: Check builtin for deprecation message and prioritize that.
  for t in [return_type] + arg_types:
    if hasattr(t, 'deprecation_message') and t.deprecation_message:
      return f'__SYCL_DEPRECATED("{t.deprecation_message}")\n'
  return ''

def get_func_prefix(builtin, return_type, arg_types):
  template_args = get_template_args(return_type, arg_types)
  func_deprecation = get_deprecation(builtin, return_type, arg_types)
  result = ""
  if template_args:
    result = result + "template <%s>\n" % (', '.join(template_args))
  if func_deprecation:
    result = result + func_deprecation
  if not template_args:
    result = result + "inline "
  return result

def generate_builtin(builtin_name, namespace, builtin, return_type, arg_types):
  func_prefix = get_func_prefix(builtin, return_type, arg_types)
  func_return = get_func_return(return_type, arg_types)
  arg_names = ["a%i" % i for i in range(len(arg_types))]
  func_args = ', '.join(["%s %s" % arg for arg in zip(arg_types, arg_names)])
  invoke = builtin.get_invoke(builtin_name, namespace, return_type, arg_types, arg_names)
  return f"""
{func_prefix}{func_return} {builtin_name}({func_args}) __NOEXC {{
{invoke}
}}
"""

def generate_builtins(builtins, namespace):
  scalar_result = []
  vector_result = []
  marray_result = []
  for builtin_name, builtin_defs in builtins.items():
    for builtin_def in builtin_defs:
      combs = type_combinations(builtin_def.return_type, builtin_def.arg_types)
      for (return_t, arg_ts) in combs:
        generated_builtin = generate_builtin(builtin_name, namespace, builtin_def, return_t, arg_ts)
        if (any([is_marray_arg(arg_t) for arg_t in arg_ts])):
          marray_result.append(generated_builtin)
        elif (any([is_vec_arg(arg_t) for arg_t in arg_ts])):
          vector_result.append(generated_builtin)
        else:
          scalar_result.append(generated_builtin)
  return (scalar_result, vector_result, marray_result)

def generate_file(directory, file_name, extra_includes, generated_builtins):
  instantiated_extra_includes = ('\n'.join([f'#include <{inc}>' for inc in extra_includes]))

  with open(os.path.join(directory, file_name), "w+") as f:
    f.write(f"""
//==--------- {file_name} - SYCL generated built-in functions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NOTE: This file was generated and should not be changed!

#pragma once

#include <sycl/detail/boolean.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

#include <sycl/builtins_utils.hpp>

{instantiated_extra_includes}

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {{
__SYCL_INLINE_VER_NAMESPACE(_V1) {{
""")

    for (namespace, builtins) in generated_builtins:
      if namespace:
        f.write(f'\nnamespace {namespace} {{')
      f.write(''.join(builtins))
      if namespace:
        f.write(f'}} // namespace {namespace}\n')

    f.write("""
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __NOEXC
""")

if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise ValueError("Invalid number of arguments! Must be given an output path.")

  scalar_builtins = []
  vector_builtins = []
  marray_builtins = []
  for (namespace, builtins) in builtins_groups:
    (sb, vb, mb) = generate_builtins(builtins, namespace)
    scalar_builtins.append((namespace, sb))
    vector_builtins.append((namespace, vb))
    marray_builtins.append((namespace, mb))

  file_path = sys.argv[1]
  generate_file(file_path, "builtins_scalar_gen.hpp", [], scalar_builtins)
  generate_file(file_path, "builtins_vector_gen.hpp", [], vector_builtins)
  generate_file(file_path, "builtins_marray_gen.hpp", ["sycl/builtins_scalar_gen.hpp", "sycl/builtins_vector_gen.hpp"], marray_builtins)
