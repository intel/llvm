import itertools
import sys

class Vec:
  def __init__(self, valid_types, valid_sizes = {1,2,3,4,8,16},
               deprecation_message=None):
    self.valid_types = valid_types
    self.valid_sizes = valid_sizes
    self.deprecation_message = deprecation_message

class InstantiatedVecArg:
  def __init__(self, template_name, vec_type):
    self.template_name = template_name
    self.vec_type = vec_type

  def __str__(self):
    return self.template_name

class InstantiatedVecReturnType:
  def __init__(self, related_arg_type, vec_type):
    self.related_arg_type = related_arg_type
    self.vec_type = vec_type

  def __str__(self):
    return f'detail::vec_return_t<{self.related_arg_type}>'

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
    if isinstance(self.referenced_type, InstantiatedVecArg) or isinstance(self.referenced_type, InstantiatedVecReturnType):
      return f'typename {self.referenced_type}::element_type'
    return self.referenced_type

class UnsignedType:
  def __init__(self, parent_idx):
    self.parent_idx = parent_idx

class InstantiatedUnsignedType:
  def __init__(self, signed_type, parent_idx):
    self.signed_type = signed_type
    self.parent_idx = parent_idx

  def __str__(self):
    return f'detail::make_unsigned_t<{self.signed_type}>'

class SameSizeIntType:
  def __init__(self, signed, parent_idx):
    self.signed = signed
    self.parent_idx = parent_idx

class InstantiatedSameSizeIntType:
  def __init__(self, parent_type, signed, parent_idx):
    self.parent_type = parent_type
    self.signed = signed
    self.parent_idx = parent_idx

  def __str__(self):
    signedness = 'signed' if self.signed else 'unsigned'
    return f'detail::same_size_{signedness}_int_t<{self.parent_type}>'

### GENTYPE DEFINITIONS
# NOTE: Marray is currently explicitly defined.

floatn = [Vec(["float"])]
vfloatn = [Vec(["float"])]
vfloat3or4 = [Vec(["float"], {3,4})]
mfloatn = []
mfloat3or4 = []
genfloatf = ["float", Vec(["float"])]

doublen = [Vec(["double"])]
vdoublen = [Vec(["double"])]
vdouble3or4 = [Vec(["double"], {3,4})]
mdoublen = []
mdouble3or4 = []
genfloatd = ["double", Vec(["double"])]

halfn = [Vec(["half"])]
vhalfn = [Vec(["half"])]
vhalf3or4 = [Vec(["half"], {3,4})]
mhalfn = []
mhalf3or4 = []
genfloath = ["half", Vec(["half"])]

genfloat = ["float", "double", "half", Vec(["float", "double", "half"])]
vgenfloat = [Vec(["float", "double", "half"])]
sgenfloat = ["float", "double", "half"]
mgenfloat = []

# NOTE: Vec size 1 is non-standard.
vgeofloat = [Vec(["float"], {1,2,3,4})]
vgeodouble = [Vec(["double"], {1,2,3,4})]
vgeohalf = [Vec(["half"], {1,2,3,4})]
gengeofloat = ["float", Vec(["float"], {1,2,3,4})]
gengeodouble = ["double", Vec(["double"], {1,2,3,4})]
gengeohalf = ["half", Vec(["half"], {1,2,3,4})]

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

mint8n = []
mint16n = []
mint32n = []
mint64n = []
muint8n = []
muint16n = []
muint32n = []
muint64n = []
mintn = []
mshortn = []
muintn = []
mulongn = []
mbooln = []

geninteger = ["char", "signed char", "short", "int", "long", "long long",
              "unsigned char", "unsigned short", "unsigned int",
              "unsigned long", "unsigned long long",
              Vec(["int8_t", "int16_t", "int32_t", "int64_t", "uint8_t",
                  "uint16_t","uint32_t","uint64_t", "long long", "unsigned long long"])]
sigeninteger = ["char", "signed char", "short", "int", "long", "long long"]
vigeninteger = [Vec(["int8_t", "int16_t", "int32_t", "int64_t", "long long"])]
migeninteger = []
igeninteger = ["char", "signed char", "short", "int", "long", "long long",
                Vec(["int8_t", "int16_t", "int32_t", "int64_t", "long long"])]
vugeninteger = [Vec(["uint8_t", "uint16_t", "uint32_t", "uint64_t", "unsigned long long"])]
sugeninteger = ["unsigned char", "unsigned short", "unsigned int",
                "unsigned long", "unsigned long long"]
ugeninteger = [Vec(["uint8_t", "uint16_t", "uint32_t", "uint64_t", "unsigned long long"]),
               "unsigned char", "unsigned short", "unsigned int",
               "unsigned long", "unsigned long long"]
igenint32 = ["int32_t", Vec(["int32_t"])]
ugenint32 = ["uint32_t", Vec(["uint32_t"])]
genint32 = ["int32_t", "uint32_t", Vec(["int32_t", "uint32_t"])]

sgentype = ["char", "signed char", "short", "int", "long", "long long",
            "unsigned char", "unsigned short", "unsigned int",
            "unsigned long", "unsigned long long", "float", "double", "half"]
vgentype = [Vec(["int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t",
                 "uint32_t", "uint64_t", "float", "double", "half", "long long",
                 "unsigned long long"])]
mgentype = []

intptr = [MultiPtr("int"), RawPtr("int")]
floatptr = [MultiPtr("float"), RawPtr("float")]
doubleptr = [MultiPtr("double"), RawPtr("double")]
halfptr = [MultiPtr("half"), RawPtr("half")]
vfloatnptr = [MultiPtr(Vec(["float"])), RawPtr(Vec(["float"]))]
vdoublenptr = [MultiPtr(Vec(["double"])), RawPtr(Vec(["double"]))]
vhalfnptr = [MultiPtr(Vec(["half"])), RawPtr(Vec(["half"]))]
vgenfloatptr = [MultiPtr(Vec(["float", "double", "half"])),
                RawPtr(Vec(["float", "double", "half"]))]
mfloatnptr = []
mdoublenptr = []
mhalfnptr = []
mintnptr = []
vint32ptr = [MultiPtr(Vec(["int32_t"])), RawPtr(Vec(["int32_t"]))]

# To help resolve template arguments, these are given the index of their parent
# argument.
elementtype0 = [ElementType(0)]
unsignedtype0 = [UnsignedType(0)]
samesizesignedint0 = [SameSizeIntType(True, 0)]
samesizeunsignedint0 = [SameSizeIntType(False, 0)]

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
  "mintn" : mintn,
  "mshortn" : mshortn,
  "muintn" : muintn,
  "mulongn" : mulongn,
  "mbooln" : mbooln,
  "geninteger" : geninteger,
  "sigeninteger" : sigeninteger,
  "vigeninteger" : vigeninteger,
  "migeninteger" : migeninteger,
  "igeninteger" : igeninteger,
  "sugeninteger" : sugeninteger,
  "vugeninteger" : vugeninteger,
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
  "mintnptr" : mintnptr,
  "vint32nptr" : vint32ptr,
  "elementtype0" : elementtype0,
  "unsignedtype0" : unsignedtype0,
  "samesizesignedint0" : samesizesignedint0,
  "samesizeunsignedint0" : samesizeunsignedint0,
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

def find_first_vec_arg(arg_types):
  for arg_type in arg_types:
    if isinstance(arg_type, InstantiatedVecArg):
      return arg_type
  return None

class Def:
  def __init__(self, name, return_type, arg_types, invoke_name=None,
               invoke_prefix="", custom_invoke=None, fast_math_invoke_name=None,
               convert_args=[], vec_size_alias=None):
    self.name = name
    self.return_type = return_type
    self.arg_types = arg_types
    if invoke_name is None:
      self.invoke_name = name
    else:
      self.invoke_name = invoke_name
    self.invoke_prefix = invoke_prefix
    self.custom_invoke = custom_invoke
    self.fast_math_invoke_name = fast_math_invoke_name
    # List of tuples with mappings for arguments to cast to argument types.
    # First element in a tuple is the index of the argument to cast and the
    # second element is the index of the argument type to convert to.
    # Alternatively, the second element can be a string representation of the
    # conversion function or type.
    self.convert_args = convert_args
    self.vec_size_alias = vec_size_alias

  def get_invoke_args(self, arg_types, arg_names):
    result = arg_names
    for (arg_idx, type_conv) in self.convert_args:
      # type_conv is either an index or a conversion function/type.
      conv = type_conv if isinstance(type_conv, str) else arg_types[type_conv]
      result[arg_idx] = f'{conv}({result[arg_idx]})'
    return ', '.join(result)

  def get_invoke(self, return_type, arg_types, arg_names):
    if self.custom_invoke:
      return self.custom_invoke(return_type, arg_types, arg_names)
    invoke_args = self.get_invoke_args(arg_types, arg_names)
    result = ""
    if self.vec_size_alias:
      vec_arg = find_first_vec_arg(arg_types)
      if vec_arg:
        result = result + f'  constexpr int {self.vec_size_alias} = detail::get_vec_size<{vec_arg.template_name}>::size;'
    if self.fast_math_invoke_name:
      result = result + f"""  if constexpr (detail::use_fast_math_v<{arg_types[0]}>)
    return __sycl_std::__invoke_{self.fast_math_invoke_name}<{return_type}>({invoke_args});\n"""
    result = result + f'  return __sycl_std::__invoke_{self.invoke_prefix}{self.invoke_name}<{return_type}>({invoke_args});'
    return result

class RelDef:
  def __init__(self, name, return_type, arg_types, invoke_name=None,
               invoke_prefix="", custom_invoke=None):
    self.name = name
    self.return_type = return_type
    self.arg_types = arg_types
    if invoke_name is None:
      self.invoke_name = name
    else:
      self.invoke_name = invoke_name
    self.invoke_prefix = invoke_prefix
    self.custom_invoke = custom_invoke

  def get_invoke(self, return_type, arg_types, arg_names):
    if self.custom_invoke:
      return self.custom_invoke(return_type, arg_types, arg_names)
    invoke_args = ', '.join(arg_names)
    return f'  return detail::RelConverter<{return_type}>::apply(__sycl_std::__invoke_{self.invoke_prefix}{self.invoke_name}<detail::internal_rel_ret_t<{return_type}>>({invoke_args}));'

def custom_signed_abs_scalar_invoke(return_type, _, arg_names):
  args = ' ,'.join(arg_names)
  return f'return static_cast<{return_type}>(__sycl_std::__invoke_s_abs<detail::make_unsigned_t<{return_type}>>({args}));'

def custom_signed_abs_vector_invoke(return_type, _, arg_names):
  args = ' ,'.join(arg_names)
  return f'return __sycl_std::__invoke_s_abs<detail::make_unsigned_t<{return_type}>>({args}).template convert<typename {return_type}::element_type>();'

def get_custom_any_all_invoke(invoke_name):
  return (lambda _, arg_types, arg_names: f"""  return detail::rel_sign_bit_test_ret_t<{arg_types[0]}>(
      __sycl_std::__invoke_{invoke_name}<detail::rel_sign_bit_test_ret_t<{arg_types[0]}>>(
          detail::rel_sign_bit_test_arg_t<{arg_types[0]}>({arg_names[0]})));""")

def custom_bool_select_invoke(return_type, _, arg_names):
  return f"""  return __sycl_std::__invoke_select<{return_type}>(
      {arg_names[0]}, {arg_names[1]}, static_cast<detail::get_select_opencl_builtin_c_arg_type<{return_type}>>({arg_names[2]}));"""

sycl_builtins = [# Math functions
                 Def("acos", "genfloat", ["genfloat"]),
                 Def("acosh", "genfloat", ["genfloat"]),
                 Def("acospi", "genfloat", ["genfloat"]),
                 Def("asin", "genfloat", ["genfloat"]),
                 Def("asinh", "genfloat", ["genfloat"]),
                 Def("asinpi", "genfloat", ["genfloat"]),
                 Def("atan", "genfloat", ["genfloat"]),
                 Def("atan2", "genfloat", ["genfloat", "genfloat"]),
                 Def("atanh", "genfloat", ["genfloat"]),
                 Def("atanpi", "genfloat", ["genfloat"]),
                 Def("atan2pi", "genfloat", ["genfloat", "genfloat"]),
                 Def("cbrt", "genfloat", ["genfloat"]),
                 Def("ceil", "genfloat", ["genfloat"]),
                 Def("copysign", "genfloat", ["genfloat", "genfloat"]),
                 Def("cos", "genfloat", ["genfloat"], fast_math_invoke_name = "native_cos"),
                 Def("cosh", "genfloat", ["genfloat"]),
                 Def("cospi", "genfloat", ["genfloat"]),
                 Def("erfc", "genfloat", ["genfloat"]),
                 Def("erf", "genfloat", ["genfloat"]),
                 Def("exp", "genfloat", ["genfloat"], fast_math_invoke_name="native_exp"),
                 Def("exp2", "genfloat", ["genfloat"], fast_math_invoke_name="native_exp2"),
                 Def("exp10", "genfloat", ["genfloat"], fast_math_invoke_name="native_exp10"),
                 Def("expm1", "genfloat", ["genfloat"]),
                 Def("fabs", "genfloat", ["genfloat"]),
                 Def("fdim", "genfloat", ["genfloat", "genfloat"]),
                 Def("floor", "genfloat", ["genfloat"]),
                 Def("fma", "genfloat", ["genfloat", "genfloat", "genfloat"]),
                 Def("fmax", "genfloat", ["genfloat", "genfloat"]),
                 Def("fmax", "vfloatn", ["vfloatn", "float"], convert_args=[(1,0)]),
                 Def("fmax", "vdoublen", ["vdoublen", "double"], convert_args=[(1,0)]),
                 Def("fmax", "vhalfn", ["vhalfn", "half"], convert_args=[(1,0)]),
                 Def("fmin", "genfloat", ["genfloat", "genfloat"]),
                 Def("fmin", "vfloatn", ["vfloatn", "float"], convert_args=[(1,0)]),
                 Def("fmin", "vdoublen", ["vdoublen", "double"], convert_args=[(1,0)]),
                 Def("fmin", "vhalfn", ["vhalfn", "half"], convert_args=[(1,0)]),
                 Def("fmod", "genfloat", ["genfloat", "genfloat"]),
                 Def("fract", "vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                 Def("fract", "float", ["float", "floatptr"]),
                 Def("fract", "double", ["double", "doubleptr"]),
                 Def("fract", "half", ["half", "halfptr"]),
                 Def("frexp", "vgenfloat", ["vgenfloat", "vint32nptr"]),
                 Def("frexp", "float", ["float", "intptr"]),
                 Def("frexp", "double", ["double", "intptr"]),
                 Def("frexp", "half", ["half", "intptr"]),
                 Def("hypot", "genfloat", ["genfloat", "genfloat"]),
                 Def("ilogb", "vint32n", ["vgenfloat"]),
                 Def("ilogb", "int", ["float"]),
                 Def("ilogb", "int", ["double"]),
                 Def("ilogb", "int", ["half"]),
                 Def("ldexp", "vgenfloat", ["vgenfloat", "vint32n"]),
                 Def("ldexp", "vgenfloat", ["vgenfloat", "int"], convert_args=[(1,"vec<int, N>")], vec_size_alias="N"),
                 Def("ldexp", "float", ["float", "int"]),
                 Def("ldexp", "double", ["double", "int"]),
                 Def("ldexp", "half", ["half", "int"]),
                 Def("lgamma", "genfloat", ["genfloat"]),
                 Def("lgamma_r", "vgenfloat", ["vgenfloat", "vint32nptr"]),
                 Def("lgamma_r", "float", ["float", "intptr"]),
                 Def("lgamma_r", "double", ["double", "intptr"]),
                 Def("lgamma_r", "half", ["half", "intptr"]),
                 Def("log", "genfloat", ["genfloat"], fast_math_invoke_name="native_log"),
                 Def("log2", "genfloat", ["genfloat"], fast_math_invoke_name="native_log2"),
                 Def("log10", "genfloat", ["genfloat"], fast_math_invoke_name="native_log10"),
                 Def("log1p", "genfloat", ["genfloat"]),
                 Def("logb", "genfloat", ["genfloat"]),
                 Def("mad", "genfloat", ["genfloat", "genfloat", "genfloat"]),
                 Def("maxmag", "genfloat", ["genfloat", "genfloat"]),
                 Def("minmag", "genfloat", ["genfloat", "genfloat"]),
                 Def("modf", "vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                 Def("modf", "float", ["float", "floatptr"]),
                 Def("modf", "double", ["double", "doubleptr"]),
                 Def("modf", "half", ["half", "halfptr"]),
                 Def("nan", "vfloatn", ["vuint32n"]),
                 Def("nan", "float", ["unsigned int"]),
                 Def("nan", "vdoublen", ["vuint64n_ext"]),
                 Def("nan", "double", ["unsigned long"]),
                 Def("nan", "double", ["unsigned long long"]),
                 Def("nan", "vhalfn", ["vuint16n"]),
                 Def("nan", "half", ["unsigned short"]),
                 Def("nextafter", "genfloat", ["genfloat", "genfloat"]),
                 Def("pow", "genfloat", ["genfloat", "genfloat"]),
                 Def("pown", "vgenfloat", ["vgenfloat", "vint32n"]),
                 Def("pown", "float", ["float", "int"]),
                 Def("pown", "double", ["double", "int"]),
                 Def("pown", "half", ["half", "int"]),
                 Def("powr", "genfloat", ["genfloat", "genfloat"], fast_math_invoke_name="native_powr"),
                 Def("remainder", "genfloat", ["genfloat", "genfloat"]),
                 Def("remquo", "vgenfloat", ["vgenfloat", "vgenfloat", "vint32nptr"]),
                 Def("remquo", "float", ["float", "float", "intptr"]),
                 Def("remquo", "double", ["double", "double", "intptr"]),
                 Def("remquo", "half", ["half", "half", "intptr"]),
                 Def("rint", "genfloat", ["genfloat"]),
                 Def("rootn", "vgenfloat", ["vgenfloat", "vint32n"]),
                 Def("rootn", "float", ["float", "int"]),
                 Def("rootn", "double", ["double", "int"]),
                 Def("rootn", "half", ["half", "int"]),
                 Def("round", "genfloat", ["genfloat"]),
                 Def("rsqrt", "genfloat", ["genfloat"], fast_math_invoke_name="native_rsqrt"),
                 Def("sin", "genfloat", ["genfloat"], fast_math_invoke_name="native_sin"),
                 Def("sincos", "vgenfloat", ["vgenfloat", "vgenfloatptr"]),
                 Def("sincos", "float", ["float", "floatptr"]),
                 Def("sincos", "vdoublen", ["vdoublen", "vdoublenptr"]),
                 Def("sincos", "double", ["double", "doubleptr"]),
                 Def("sincos", "vhalfn", ["vhalfn", "vhalfnptr"]),
                 Def("sincos", "half", ["half", "halfptr"]),
                 Def("sinh", "genfloat", ["genfloat"]),
                 Def("sinpi", "genfloat", ["genfloat"]),
                 Def("sqrt", "genfloat", ["genfloat"], fast_math_invoke_name="native_sqrt"),
                 Def("tan", "genfloat", ["genfloat"], fast_math_invoke_name="native_tan"),
                 Def("tanh", "genfloat", ["genfloat"]),
                 Def("tanpi", "genfloat", ["genfloat"]),
                 Def("tgamma", "genfloat", ["genfloat"]),
                 Def("trunc", "genfloat", ["genfloat"]),
                 # Integer functions
                 Def("abs", "sigeninteger", ["sigeninteger"], custom_invoke=custom_signed_abs_scalar_invoke),
                 Def("abs", "vigeninteger", ["vigeninteger"], custom_invoke=custom_signed_abs_vector_invoke),
                 Def("abs", "ugeninteger", ["ugeninteger"], invoke_prefix="u_"),
                 Def("abs_diff", "unsignedtype0", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("abs_diff", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("add_sat", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("add_sat", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("hadd", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("hadd", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("rhadd", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("rhadd", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("clamp", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("clamp", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("clamp", "vigeninteger", ["vigeninteger", "elementtype0", "elementtype0"], invoke_prefix="s_"),
                 Def("clamp", "vugeninteger", ["vugeninteger", "elementtype0", "elementtype0"], invoke_prefix="u_"),
                 Def("clz", "geninteger", ["geninteger"]),
                 Def("ctz", "geninteger", ["geninteger"]),
                 Def("mad_hi", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("mad_hi", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("mad_sat", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("mad_sat", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("(max)", "igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_max"),
                 Def("(max)", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_max"),
                 Def("(max)", "vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_max"),
                 Def("(max)", "vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_max"),
                 Def("(min)", "igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_min"),
                 Def("(min)", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_min"),
                 Def("(min)", "vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_min"),
                 Def("(min)", "vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_min"),
                 Def("mul_hi", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("mul_hi", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("rotate", "geninteger", ["geninteger", "geninteger"]),
                 Def("sub_sat", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("sub_sat", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("upsample", "int16_t", ["int8_t", "uint8_t"], invoke_prefix="s_"),
                 Def("upsample", "vint16n", ["vint8n", "vuint8n"], invoke_prefix="s_"),
                 Def("upsample", "uint16_t", ["uint8_t", "uint8_t"], invoke_prefix="u_"),
                 Def("upsample", "vuint16n", ["vuint8n", "vuint8n"], invoke_prefix="u_"),
                 Def("upsample", "int32_t", ["int16_t", "uint16_t"], invoke_prefix="s_"),
                 Def("upsample", "vint32n", ["vint16n", "vuint16n"], invoke_prefix="s_"),
                 Def("upsample", "uint32_t", ["uint16_t", "uint16_t"], invoke_prefix="u_"),
                 Def("upsample", "vuint32n", ["vuint16n", "vuint16n"], invoke_prefix="u_"),
                 Def("upsample", "int64_t", ["int32_t", "uint32_t"], invoke_prefix="s_"),
                 Def("upsample", "vint64n", ["vint32n", "vuint32n"], invoke_prefix="s_"),
                 Def("upsample", "uint64_t", ["uint32_t", "uint32_t"], invoke_prefix="u_"),
                 Def("upsample", "vuint64n", ["vuint32n", "vuint32n"], invoke_prefix="u_"),
                 Def("popcount", "geninteger", ["geninteger"]),
                 Def("mad24", "igenint32", ["igenint32", "igenint32", "igenint32"], invoke_prefix="s_"),
                 Def("mad24", "ugenint32", ["ugenint32", "ugenint32", "ugenint32"], invoke_prefix="u_"),
                 Def("mul24", "igenint32", ["igenint32", "igenint32"], invoke_prefix="s_"),
                 Def("mul24", "ugenint32", ["ugenint32", "ugenint32"], invoke_prefix="u_"),
                 # Common functions
                 Def("clamp", "genfloat", ["genfloat", "genfloat", "genfloat"], invoke_prefix="f"),
                 Def("clamp", "vfloatn", ["vfloatn", "float", "float"], invoke_prefix="f", convert_args=[(1,0),(2,0)]),
                 Def("clamp", "vdoublen", ["vdoublen", "double", "double"], invoke_prefix="f", convert_args=[(1,0),(2,0)]),
                 Def("degrees", "genfloat", ["genfloat"]),
                 Def("(max)", "genfloat", ["genfloat", "genfloat"], invoke_name="fmax_common"),
                 Def("(max)", "vfloatn", ["vfloatn", "float"], invoke_name="fmax_common", convert_args=[(1,0)]),
                 Def("(max)", "vdoublen", ["vdoublen", "double"], invoke_name="fmax_common", convert_args=[(1,0)]),
                 Def("(min)", "genfloat", ["genfloat", "genfloat"], invoke_name="fmin_common"),
                 Def("(min)", "vfloatn", ["vfloatn", "float"], invoke_name="fmin_common", convert_args=[(1,0)]),
                 Def("(min)", "vdoublen", ["vdoublen", "double"], invoke_name="fmin_common", convert_args=[(1,0)]),
                 Def("mix", "genfloat", ["genfloat", "genfloat", "genfloat"]),
                 Def("mix", "vfloatn", ["vfloatn", "vfloatn", "float"], convert_args=[(2,0)]),
                 Def("mix", "vdoublen", ["vdoublen", "vdoublen", "double"], convert_args=[(2,0)]),
                 Def("radians", "genfloat", ["genfloat"]),
                 Def("step", "genfloat", ["genfloat", "genfloat"]),
                 Def("step", "vfloatn", ["float", "vfloatn"], convert_args=[(0,1)]),
                 Def("step", "vdoublen", ["double", "vdoublen"], convert_args=[(0,1)]),
                 Def("smoothstep", "genfloat", ["genfloat", "genfloat", "genfloat"]),
                 Def("smoothstep", "vfloatn", ["float", "float", "vfloatn"], convert_args=[(0,2),(1,2)]),
                 Def("smoothstep", "vdoublen", ["double", "double", "vdoublen"], convert_args=[(0,2),(1,2)]),
                 Def("sign", "genfloat", ["genfloat"]),
                 Def("abs", "genfloat", ["genfloat"], invoke_prefix="f"), # TODO: Non-standard. Deprecate.
                 # Geometric functions
                 Def("cross", "vfloat3or4", ["vfloat3or4", "vfloat3or4"]),
                 Def("cross", "vdouble3or4", ["vdouble3or4", "vdouble3or4"]),
                 Def("cross", "vhalf3or4", ["vhalf3or4", "vhalf3or4"]), # TODO: Non-standard. Deprecate.
                 Def("dot", "float", ["vgeofloat", "vgeofloat"], invoke_name="Dot"),
                 Def("dot", "double", ["vgeodouble", "vgeodouble"], invoke_name="Dot"),
                 Def("dot", "half", ["vgeohalf", "vgeohalf"], invoke_name="Dot"), # TODO: Non-standard. Deprecate.
                 Def("dot", "sgenfloat", ["sgenfloat", "sgenfloat"], custom_invoke=(lambda return_types, arg_types, arg_names: '  return ' + ' * '.join(arg_names) + ';')),
                 Def("distance", "float", ["gengeofloat", "gengeofloat"]),
                 Def("distance", "double", ["gengeodouble", "gengeodouble"]),
                 Def("distance", "half", ["gengeohalf", "gengeohalf"]), # TODO: Non-standard. Deprecate.
                 Def("length", "float", ["gengeofloat"]),
                 Def("length", "double", ["gengeodouble"]),
                 Def("length", "half", ["gengeohalf"]), # TODO: Non-standard. Deprecate.
                 Def("normalize", "gengeofloat", ["gengeofloat"]),
                 Def("normalize", "gengeodouble", ["gengeodouble"]),
                 Def("normalize", "gengeohalf", ["gengeohalf"]), # TODO: Non-standard. Deprecate.
                 Def("fast_distance", "float", ["gengeofloat", "gengeofloat"]),
                 Def("fast_distance", "double", ["gengeodouble", "gengeodouble"]),
                 Def("fast_length", "float", ["gengeofloat"]),
                 Def("fast_length", "double", ["gengeodouble"]),
                 Def("fast_normalize", "gengeofloat", ["gengeofloat"]),
                 Def("fast_normalize", "gengeodouble", ["gengeodouble"]),
                 # Relational functions
                 RelDef("isequal", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdEqual"),
                 RelDef("isequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdEqual"),
                 RelDef("isnotequal", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FUnordNotEqual"),
                 RelDef("isnotequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FUnordNotEqual"),
                 RelDef("isgreater", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreater", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreaterequal", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isgreaterequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isless", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdLessThan"),
                 RelDef("isless", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThan"),
                 RelDef("islessequal", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessgreater", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdNotEqual"),
                 RelDef("islessgreater", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdNotEqual"),
                 RelDef("isfinite", "samesizesignedint0", ["vgenfloat"], invoke_name="IsFinite"),
                 RelDef("isfinite", "bool", ["sgenfloat"], invoke_name="IsFinite"),
                 RelDef("isinf", "samesizesignedint0", ["vgenfloat"], invoke_name="IsInf"),
                 RelDef("isinf", "bool", ["sgenfloat"], invoke_name="IsInf"),
                 RelDef("isnan", "samesizesignedint0", ["vgenfloat"], invoke_name="IsNan"),
                 RelDef("isnan", "bool", ["sgenfloat"], invoke_name="IsNan"),
                 RelDef("isnormal", "samesizesignedint0", ["vgenfloat"], invoke_name="IsNormal"),
                 RelDef("isnormal", "bool", ["sgenfloat"], invoke_name="IsNormal"),
                 RelDef("isordered", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Ordered"),
                 RelDef("isordered", "bool", ["sgenfloat", "sgenfloat"], invoke_name="Ordered"),
                 RelDef("isunordered", "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Unordered"),
                 RelDef("isunordered", "bool", ["sgenfloat", "sgenfloat"], invoke_name="Unordered"),
                 RelDef("signbit", "samesizesignedint0", ["vgenfloat"], invoke_name="SignBitSet"),
                 RelDef("signbit", "bool", ["sgenfloat"], invoke_name="SignBitSet"),
                 Def("any", "int", ["vigeninteger"], custom_invoke=get_custom_any_all_invoke("Any")),
                 Def("any", "bool", ["sigeninteger"], custom_invoke=(lambda return_type, arg_types, arg_names: f'  return detail::Boolean<1>(int(detail::msbIsSet({arg_names[0]})));')),
                 Def("all", "int", ["vigeninteger"], custom_invoke=get_custom_any_all_invoke("All")),
                 Def("all", "bool", ["sigeninteger"], custom_invoke=(lambda return_type, arg_types, arg_names: f'  return detail::Boolean<1>(int(detail::msbIsSet({arg_names[0]})));')),
                 Def("bitselect", "vgentype", ["vgentype", "vgentype", "vgentype"]),
                 Def("bitselect", "sgentype", ["sgentype", "sgentype", "sgentype"]),
                 Def("select", "vint8n", ["vint8n", "vint8n", "vint8n"]),
                 Def("select", "vint16n", ["vint16n", "vint16n", "vint16n"]),
                 Def("select", "vint32n", ["vint32n", "vint32n", "vint32n"]),
                 Def("select", "vint64n_ext", ["vint64n_ext", "vint64n_ext", "vint64n_ext"]),
                 Def("select", "vuint8n", ["vuint8n", "vuint8n", "vint8n"]),
                 Def("select", "vuint16n", ["vuint16n", "vuint16n", "vint16n"]),
                 Def("select", "vuint32n", ["vuint32n", "vuint32n", "vint32n"]),
                 Def("select", "vuint64n_ext", ["vuint64n_ext", "vuint64n_ext", "vint64n_ext"]),
                 Def("select", "vfloatn", ["vfloatn", "vfloatn", "vint32n"]),
                 Def("select", "vdoublen", ["vdoublen", "vdoublen", "vint64n_ext"]),
                 Def("select", "vhalfn", ["vhalfn", "vhalfn", "vint16n"]),
                 Def("select", "vint8n", ["vint8n", "vint8n", "vuint8n"]),
                 Def("select", "vint16n", ["vint16n", "vint16n", "vuint16n"]),
                 Def("select", "vint32n", ["vint32n", "vint32n", "vuint32n"]),
                 Def("select", "vint64n_ext", ["vint64n_ext", "vint64n_ext", "vuint64n_ext"]),
                 Def("select", "vuint8n", ["vuint8n", "vuint8n", "vuint8n"]),
                 Def("select", "vuint16n", ["vuint16n", "vuint16n", "vuint16n"]),
                 Def("select", "vuint32n", ["vuint32n", "vuint32n", "vuint32n"]),
                 Def("select", "vuint64n_ext", ["vuint64n_ext", "vuint64n_ext", "vuint64n_ext"]),
                 Def("select", "vfloatn", ["vfloatn", "vfloatn", "vuint32n"]),
                 Def("select", "vdoublen", ["vdoublen", "vdoublen", "vuint64n_ext"]),
                 Def("select", "vhalfn", ["vhalfn", "vhalfn", "vuint16n"]),
                 Def("select", "sgentype", ["sgentype", "sgentype", "bool"], custom_invoke=custom_bool_select_invoke)]
native_builtins = [Def("cos", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("divide", "genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_"),
                   Def("exp", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("exp2", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("exp10", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("log", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("log2", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("log10", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("powr", "genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_"),
                   Def("recip", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("rsqrt", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("sin", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("sqrt", "genfloatf", ["genfloatf"], invoke_prefix="native_"),
                   Def("tan", "genfloatf", ["genfloatf"], invoke_prefix="native_")]
half_precision_builtins = [Def("cos", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("divide", "genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_"),
                           Def("exp", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("exp2", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("exp10", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("log", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("log2", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("log10", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("powr", "genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_"),
                           Def("recip", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("rsqrt", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("sin", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("sqrt", "genfloatf", ["genfloatf"], invoke_prefix="half_"),
                           Def("tan", "genfloatf", ["genfloatf"], invoke_prefix="half_")]

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
  if isinstance(mapping, UnsignedType):
    parent_mapping = mappings[arg_types[mapping.parent_idx]]
    return InstantiatedUnsignedType(parent_mapping, mapping.parent_idx)
  if isinstance(mapping, SameSizeIntType):
    parent_mapping = mappings[arg_types[mapping.parent_idx]]
    return InstantiatedSameSizeIntType(parent_mapping, mapping.signed, mapping.parent_idx)
  return mapping

def instantiate_arg(idx, arg):
  if isinstance(arg, Vec):
    return InstantiatedVecArg(f'T{idx}', arg)
  if isinstance(arg, MultiPtr):
    return MultiPtr(instantiate_arg(idx, arg.element_type))
  if isinstance(arg, RawPtr):
    return RawPtr(instantiate_arg(idx, arg.element_type))
  if isinstance(arg, InstantiatedElementType):
    return InstantiatedElementType(instantiate_arg(arg.parent_idx, arg.referenced_type), arg.parent_idx)
  if isinstance(arg, InstantiatedUnsignedType):
    return InstantiatedUnsignedType(instantiate_arg(arg.parent_idx, arg.signed_type), arg.parent_idx)
  if isinstance(arg, InstantiatedSameSizeIntType):
    return InstantiatedSameSizeIntType(instantiate_arg(arg.parent_idx, arg.parent_type), arg.signed, arg.parent_idx)
  return arg

def instantiate_return_type(return_type, instantiated_args):
  if isinstance(return_type, Vec):
    first_vec = find_first_vec_arg(instantiated_args)
    return InstantiatedVecReturnType(str(first_vec), return_type)
  if isinstance(return_type, MultiPtr):
    return MultiPtr(instantiate_return_type(return_type.element_type, instantiated_args))
  if isinstance(return_type, RawPtr):
    return RawPtr(instantiate_return_type(return_type.element_type, instantiated_args))
  if isinstance(return_type, InstantiatedElementType):
    return InstantiatedElementType(instantiate_return_type(return_type.referenced_type, instantiated_args), return_type.parent_idx)
  if isinstance(return_type, InstantiatedUnsignedType):
    return InstantiatedUnsignedType(instantiate_return_type(return_type.signed_type, instantiated_args), return_type.parent_idx)
  if isinstance(return_type, InstantiatedSameSizeIntType):
    return InstantiatedSameSizeIntType(instantiate_return_type(return_type.parent_type, instantiated_args), return_type.signed, return_type.parent_idx)
  return return_type

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
    result.append((instantiated_return_type, instantiated_arg_types))
  return result

def any_vector(types):
  for t in types:
    if isinstance(t, Vec):
      return True
  return False

def any_multi_ptr(types):
  for t in types:
    if isinstance(t, MultiPtr):
      return True
  return False

def get_all_vec_args(arg_types):
  result = []
  for arg_type in arg_types:
    if isinstance(arg_type, InstantiatedVecArg):
      result.append(arg_type)
    if isinstance(arg_type, MultiPtr) or isinstance(arg_type, RawPtr):
      result = result + get_all_vec_args([arg_type.element_type])
  return result

def get_vec_arg_requirement(vec_arg):
  valid_type_str = ', '.join(vec_arg.vec_type.valid_types)
  valid_sizes_str = ', '.join(map(str, vec_arg.vec_type.valid_sizes))
  checks = [f'detail::is_vec_v<{vec_arg}>',
            f'detail::is_valid_vec_type_v<{vec_arg.template_name}, {valid_type_str}>',
            f'detail::is_valid_vec_size_v<{vec_arg.template_name}, {valid_sizes_str}>']
  return '(' + (' && '.join(checks)) + ')'

def get_func_return(return_type, arg_types):
  vec_args = get_all_vec_args(arg_types)
  if len(vec_args) > 0:
    conjunc_reqs = ' && '.join([get_vec_arg_requirement(vec_arg) for vec_arg in vec_args])
    return f'std::enable_if_t<{conjunc_reqs}, {return_type}>'
  return str(return_type)

def get_template_args(return_type, arg_types):
  vec_args = get_all_vec_args(arg_types)
  result = [f'typename {vec_arg.template_name}' for vec_arg in vec_args]
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

def generate_builtin(builtin, return_type, arg_types):
  func_prefix = get_func_prefix(builtin, return_type, arg_types)
  func_return = get_func_return(return_type, arg_types)
  arg_names = ["a%i" % i for i in range(len(arg_types))]
  func_args = ', '.join(["%s %s" % arg for arg in zip(arg_types, arg_names)])
  invoke = builtin.get_invoke(return_type, arg_types, arg_names)
  return f"""
{func_prefix}{func_return} {builtin.name}({func_args}) __NOEXC {{
{invoke}
}}
"""

def generate_builtins(builtins):
  result = []
  for builtin in builtins:
    for (return_t, arg_ts) in type_combinations(builtin.return_type, builtin.arg_types):
      result.append(generate_builtin(builtin, return_t, arg_ts))
  return result

if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise ValueError("Invalid number of arguments! Must be given an output file.")

  with open(sys.argv[1], "w+") as f:
    f.write("""
//==--------- builtins_gen.hpp - SYCL generated built-in functions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file was generated and should not be changed!

#pragma once

#include <sycl/detail/boolean.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

#include <sycl/builtins_utils.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

""")

    for (namespace, builtins) in builtins_groups:
      if namespace:
        f.write(f'\nnamespace {namespace} {{')
      f.write(''.join(generate_builtins(builtins)))
      if namespace:
        f.write(f'}} // namespace {namespace}\n')

    f.write("""
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __NOEXC
""")
