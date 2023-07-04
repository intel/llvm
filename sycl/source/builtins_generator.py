import itertools
import sys

class Vec:
  def __init__(self, element_type, valid_sizes = [1,2,3,4,8,16]):
    self.element_type = element_type
    self.valid_sizes = valid_sizes

  def __str__(self):
    return f'vec<{self.element_type}, N>'

class MultiPtr:
  def __init__(self, element_type):
    self.element_type = element_type

  def __str__(self):
    return f'multi_ptr<{self.element_type}, Space, IsDecorated>'

# TODO: Raw pointer variants kept for compatibility. They should be deprecated.
class RawPtr:
  def __init__(self, element_type):
    self.element_type = element_type

  def __str__(self):
    return f'{self.element_type}*'

class ElementType:
  pass

class InstantiatedElementType:
  def __init__(self, referenced_type):
    self.referenced_type = referenced_type

  def __str__(self):
    if isinstance(self.referenced_type, Vec):
      return self.referenced_type.element_type
    return self.referenced_type

class UnsignedType:
  pass

class InstantiatedUnsignedType:
  def __init__(self, signed_type):
    self.signed_type = signed_type

  def __str__(self):
    return f'detail::make_unsigned_t<{self.signed_type}>'

### GENTYPE DEFINITIONS
# NOTE: Marray is currently explicitly defined.

floatn = [Vec("float")]
vfloatn = [Vec("float")]
vfloat3or4 = [Vec("float", [3,4])]
mfloatn = []
mfloat3or4 = []
genfloatf = ["float", Vec("float")]

doublen = [Vec("double")]
vdoublen = [Vec("double")]
vdouble3or4 = [Vec("double", [3,4])]
mdoublen = []
mdouble3or4 = []
genfloatd = ["double", Vec("double")]

halfn = [Vec("half")]
vhalfn = [Vec("half")]
vhalf3or4 = [Vec("half", [3,4])]
mhalfn = []
mhalf3or4 = []
genfloath = ["half", Vec("half")]

genfloat = ["float", "double", "half", Vec("float"), Vec("double"), Vec("half")]
sgenfloat = ["float", "double", "half"]
mgenfloat = []

vgeofloat = [Vec("float", [2,3,4])]
vgeodouble = [Vec("double", [2,3,4])]
vgeohalf = [Vec("half", [2,3,4])]
gengeofloat = ["float", Vec("float", [2,3,4])]
gengeodouble = ["double", Vec("double", [2,3,4])]
gengeohalf = ["half", Vec("half", [2,3,4])]

vint8n = [Vec("int8_t")]
vint16n = [Vec("int16_t")]
vint32n = [Vec("int32_t")]
vint64n = [Vec("int64_t")]
vuint8n = [Vec("uint8_t")]
vuint16n = [Vec("uint16_t")]
vuint32n = [Vec("uint32_t")]
vuint64n = [Vec("uint64_t")]

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
              Vec("int8_t"), Vec("int16_t"), Vec("int32_t"), Vec("int64_t"),
              Vec("uint8_t"), Vec("uint16_t"), Vec("uint32_t"), Vec("uint64_t")]
sigeninteger = ["char", "signed char", "short", "int", "long", "long long"]
vigeninteger = [Vec("int8_t"), Vec("int16_t"), Vec("int32_t"), Vec("int64_t")]
migeninteger = []
igeninteger = ["char", "signed char", "short", "int", "long", "long long",
                Vec("int8_t"), Vec("int16_t"), Vec("int32_t"), Vec("int64_t")]
vugeninteger = [Vec("uint8_t"), Vec("uint16_t"), Vec("uint32_t"), Vec("uint64_t")]
sugeninteger = ["unsigned char", "unsigned short", "unsigned int",
                "unsigned long", "unsigned long long"]
ugeninteger = [Vec("uint8_t"), Vec("uint16_t"), Vec("uint32_t"), Vec("uint64_t"),
               "unsigned char", "unsigned short", "unsigned int",
               "unsigned long", "unsigned long long"]
igenint32 = ["int32_t", Vec("int32_t")]
ugenint32 = ["uint32_t", Vec("uint32_t")]
genint32 = ["int32_t", "uint32_t", Vec("int32_t"), Vec("uint32_t")]

sgentype = ["char", "signed char", "short", "int", "long", "long long",
            "unsigned char", "unsigned short", "unsigned int",
            "unsigned long", "unsigned long long", "float", "double", "half"]
vgentype = [Vec("int8_t"), Vec("int16_t"), Vec("int32_t"), Vec("int64_t"),
            Vec("uint8_t"), Vec("uint16_t"), Vec("uint32_t"), Vec("uint64_t"),
            Vec("float"), Vec("double"), Vec("half")]
mgentype = []

intptr = [MultiPtr("int"), RawPtr("int")]
floatptr = [MultiPtr("float"), RawPtr("float")]
doubleptr = [MultiPtr("double"), RawPtr("double")]
halfptr = [MultiPtr("half"), RawPtr("half")]
vfloatnptr = [MultiPtr(Vec("float")), RawPtr(Vec("float"))]
vdoublenptr = [MultiPtr(Vec("double")), RawPtr(Vec("double"))]
vhalfnptr = [MultiPtr(Vec("half")), RawPtr(Vec("half"))]
mfloatnptr = []
mdoublenptr = []
mhalfnptr = []
mintnptr = []
vint32ptr = [MultiPtr(Vec("int32_t")), RawPtr(Vec("int32_t"))]

elementtype = [ElementType()]
unsignedtype = [UnsignedType()]

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
  "vuint8n" : vuint8n,
  "vuint16n" : vuint16n,
  "vuint32n" : vuint32n,
  "vuint64n" : vuint64n,
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
  "mfloatnptr" : mfloatnptr,
  "mdoublenptr" : mdoublenptr,
  "mhalfnptr" : mhalfnptr,
  "mintnptr" : mintnptr,
  "vint32nptr" : vint32ptr,
  "elementtype" : elementtype,
  "unsignedtype" : unsignedtype,
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

class Def:
  def __init__(self, name, return_type, arg_types, invoke_name=None,
               invoke_prefix="", custom_invoke=None, fast_math_invoke_name=None,
               convert_args=[]):
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
  return f'return __sycl_std::__invoke_s_abs<detail::make_unsigned_t<{return_type}>>({args}).template convert<{return_type.element_type}>();'

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
                 Def("fract", "vfloatn", ["vfloatn", "vfloatnptr"]),
                 Def("fract", "float", ["float", "floatptr"]),
                 Def("fract", "vdoublen", ["vdoublen", "vdoublenptr"]),
                 Def("fract", "double", ["double", "doubleptr"]),
                 Def("fract", "vhalfn", ["vhalfn", "vhalfnptr"]),
                 Def("fract", "half", ["half", "halfptr"]),
                 Def("frexp", "vfloatn", ["vfloatn", "vint32nptr"]),
                 Def("frexp", "float", ["float", "intptr"]),
                 Def("frexp", "vdoublen", ["vdoublen", "vint32nptr"]),
                 Def("frexp", "double", ["double", "intptr"]),
                 Def("frexp", "vhalfn", ["vhalfn", "vint32nptr"]),
                 Def("frexp", "half", ["half", "intptr"]),
                 Def("hypot", "genfloat", ["genfloat", "genfloat"]),
                 Def("ilogb", "vint32n", ["vfloatn"]),
                 Def("ilogb", "int", ["float"]),
                 Def("ilogb", "vint32n", ["vdoublen"]),
                 Def("ilogb", "int", ["double"]),
                 Def("ilogb", "vint32n", ["vhalfn"]),
                 Def("ilogb", "int", ["half"]),
                 Def("ldexp", "vfloatn", ["vfloatn", "vint32n"]),
                 Def("ldexp", "floatn", ["floatn", "int"], convert_args=[(1,"vec<int, N>")]),
                 Def("ldexp", "float", ["float", "int"]),
                 Def("ldexp", "vdoublen", ["vdoublen", "vint32n"]),
                 Def("ldexp", "doublen", ["doublen", "int"], convert_args=[(1,"vec<int, N>")]),
                 Def("ldexp", "double", ["double", "int"]),
                 Def("ldexp", "vhalfn", ["vhalfn", "vint32n"]),
                 Def("ldexp", "halfn", ["halfn", "int"], convert_args=[(1,"vec<int, N>")]),
                 Def("ldexp", "half", ["half", "int"]),
                 Def("lgamma", "genfloat", ["genfloat"]),
                 Def("lgamma_r", "vfloatn", ["vfloatn", "vint32nptr"]),
                 Def("lgamma_r", "float", ["float", "intptr"]),
                 Def("lgamma_r", "vdoublen", ["vdoublen", "vint32nptr"]),
                 Def("lgamma_r", "double", ["double", "intptr"]),
                 Def("lgamma_r", "vhalfn", ["vhalfn", "vint32nptr"]),
                 Def("lgamma_r", "half", ["half", "intptr"]),
                 Def("log", "genfloat", ["genfloat"], fast_math_invoke_name="native_log"),
                 Def("log2", "genfloat", ["genfloat"], fast_math_invoke_name="native_log2"),
                 Def("log10", "genfloat", ["genfloat"], fast_math_invoke_name="native_log10"),
                 Def("log1p", "genfloat", ["genfloat"]),
                 Def("logb", "genfloat", ["genfloat"]),
                 Def("mad", "genfloat", ["genfloat", "genfloat", "genfloat"]),
                 Def("maxmag", "genfloat", ["genfloat", "genfloat"]),
                 Def("minmag", "genfloat", ["genfloat", "genfloat"]),
                 Def("modf", "vfloatn", ["vfloatn", "vfloatnptr"]),
                 Def("modf", "float", ["float", "floatptr"]),
                 Def("modf", "vdoublen", ["vdoublen", "vdoublenptr"]),
                 Def("modf", "double", ["double", "doubleptr"]),
                 Def("modf", "vhalfn", ["vhalfn", "vhalfnptr"]),
                 Def("modf", "half", ["half", "halfptr"]),
                 Def("nan", "vfloatn", ["vuint32n"]),
                 Def("nan", "float", ["unsigned int"]),
                 Def("nan", "vdoublen", ["vuint64n"]),
                 Def("nan", "double", ["unsigned long"]),
                 Def("nan", "double", ["unsigned long long"]),
                 Def("nan", "vhalfn", ["vuint16n"]),
                 Def("nan", "half", ["unsigned short"]),
                 Def("nextafter", "genfloat", ["genfloat", "genfloat"]),
                 Def("pow", "genfloat", ["genfloat", "genfloat"]),
                 Def("pown", "vfloatn", ["vfloatn", "vint32n"]),
                 Def("pown", "float", ["float", "int"]),
                 Def("pown", "vdoublen", ["vdoublen", "vint32n"]),
                 Def("pown", "double", ["double", "int"]),
                 Def("pown", "vhalfn", ["vhalfn", "vint32n"]),
                 Def("pown", "half", ["half", "int"]),
                 Def("powr", "genfloat", ["genfloat", "genfloat"], fast_math_invoke_name="native_powr"),
                 Def("remainder", "genfloat", ["genfloat", "genfloat"]),
                 Def("remquo", "vfloatn", ["vfloatn", "vfloatn", "vint32nptr"]),
                 Def("remquo", "float", ["float", "float", "intptr"]),
                 Def("remquo", "vdoublen", ["vdoublen", "vdoublen", "vint32nptr"]),
                 Def("remquo", "double", ["double", "double", "intptr"]),
                 Def("remquo", "vhalfn", ["vhalfn", "vhalfn", "vint32nptr"]),
                 Def("remquo", "half", ["half", "half", "intptr"]),
                 Def("rint", "genfloat", ["genfloat"]),
                 Def("rootn", "vfloatn", ["vfloatn", "vint32n"]),
                 Def("rootn", "float", ["float", "int"]),
                 Def("rootn", "vdoublen", ["vdoublen", "vint32n"]),
                 Def("rootn", "double", ["double", "int"]),
                 Def("rootn", "vhalfn", ["vhalfn", "vint32n"]),
                 Def("rootn", "half", ["half", "int"]),
                 Def("round", "genfloat", ["genfloat"]),
                 Def("rsqrt", "genfloat", ["genfloat"], fast_math_invoke_name="native_rsqrt"),
                 Def("sin", "genfloat", ["genfloat"], fast_math_invoke_name="native_sin"),
                 Def("sincos", "vfloatn", ["vfloatn", "vfloatnptr"]),
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
                 Def("abs_diff", "unsignedtype", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("abs_diff", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("add_sat", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("add_sat", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("hadd", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("hadd", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("rhadd", "igeninteger", ["igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("rhadd", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("clamp", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("clamp", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("clamp", "vigeninteger", ["vigeninteger", "elementtype", "elementtype"], invoke_prefix="s_"),
                 Def("clamp", "vugeninteger", ["vugeninteger", "elementtype", "elementtype"], invoke_prefix="u_"),
                 Def("clz", "geninteger", ["geninteger"]),
                 Def("ctz", "geninteger", ["geninteger"]),
                 Def("mad_hi", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("mad_hi", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("mad_sat", "igeninteger", ["igeninteger", "igeninteger", "igeninteger"], invoke_prefix="s_"),
                 Def("mad_sat", "ugeninteger", ["ugeninteger", "ugeninteger", "ugeninteger"], invoke_prefix="u_"),
                 Def("(max)", "igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_max"),
                 Def("(max)", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_max"),
                 Def("(max)", "vigeninteger", ["vigeninteger", "elementtype"], invoke_name="s_max"),
                 Def("(max)", "vugeninteger", ["vugeninteger", "elementtype"], invoke_name="u_max"),
                 Def("(min)", "igeninteger", ["igeninteger", "igeninteger"], invoke_name="s_min"),
                 Def("(min)", "ugeninteger", ["ugeninteger", "ugeninteger"], invoke_name="u_min"),
                 Def("(min)", "vigeninteger", ["vigeninteger", "elementtype"], invoke_name="s_min"),
                 Def("(min)", "vugeninteger", ["vugeninteger", "elementtype"], invoke_name="u_min"),
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
                 RelDef("isequal", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdEqual"),
                 RelDef("isequal", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdEqual"),
                 RelDef("isequal", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdEqual"),
                 RelDef("isequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdEqual"),
                 RelDef("isnotequal", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FUnordNotEqual"),
                 RelDef("isnotequal", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FUnordNotEqual"),
                 RelDef("isnotequal", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FUnordNotEqual"),
                 RelDef("isnotequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FUnordNotEqual"),
                 RelDef("isgreater", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreater", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreater", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreater", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThan"),
                 RelDef("isgreaterequal", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isgreaterequal", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isgreaterequal", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isgreaterequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThanEqual"),
                 RelDef("isless", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdLessThan"),
                 RelDef("isless", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdLessThan"),
                 RelDef("isless", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdLessThan"),
                 RelDef("isless", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThan"),
                 RelDef("islessequal", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessequal", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessequal", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessequal", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThanEqual"),
                 RelDef("islessgreater", "vint16n", ["vhalfn", "vhalfn"], invoke_name="FOrdNotEqual"),
                 RelDef("islessgreater", "vint32n", ["vfloatn", "vfloatn"], invoke_name="FOrdNotEqual"),
                 RelDef("islessgreater", "vint64n", ["vdoublen", "vdoublen"], invoke_name="FOrdNotEqual"),
                 RelDef("islessgreater", "bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdNotEqual"),
                 RelDef("isfinite", "vint16n", ["vhalfn"], invoke_name="IsFinite"),
                 RelDef("isfinite", "vint32n", ["vfloatn"], invoke_name="IsFinite"),
                 RelDef("isfinite", "vint64n", ["vdoublen"], invoke_name="IsFinite"),
                 RelDef("isfinite", "bool", ["sgenfloat"], invoke_name="IsFinite"),
                 RelDef("isinf", "vint16n", ["vhalfn"], invoke_name="IsInf"),
                 RelDef("isinf", "vint32n", ["vfloatn"], invoke_name="IsInf"),
                 RelDef("isinf", "vint64n", ["vdoublen"], invoke_name="IsInf"),
                 RelDef("isinf", "bool", ["sgenfloat"], invoke_name="IsInf"),
                 RelDef("isnan", "vint16n", ["vhalfn"], invoke_name="IsNan"),
                 RelDef("isnan", "vint32n", ["vfloatn"], invoke_name="IsNan"),
                 RelDef("isnan", "vint64n", ["vdoublen"], invoke_name="IsNan"),
                 RelDef("isnan", "bool", ["sgenfloat"], invoke_name="IsNan"),
                 RelDef("isnormal", "vint16n", ["vhalfn"], invoke_name="IsNormal"),
                 RelDef("isnormal", "vint32n", ["vfloatn"], invoke_name="IsNormal"),
                 RelDef("isnormal", "vint64n", ["vdoublen"], invoke_name="IsNormal"),
                 RelDef("isnormal", "bool", ["sgenfloat"], invoke_name="IsNormal"),
                 RelDef("isordered", "vint16n", ["vhalfn", "vhalfn"], invoke_name="Ordered"),
                 RelDef("isordered", "vint32n", ["vfloatn", "vfloatn"], invoke_name="Ordered"),
                 RelDef("isordered", "vint64n", ["vdoublen", "vdoublen"], invoke_name="Ordered"),
                 RelDef("isordered", "bool", ["sgenfloat", "sgenfloat"], invoke_name="Ordered"),
                 RelDef("isunordered", "vint16n", ["vhalfn", "vhalfn"], invoke_name="Unordered"),
                 RelDef("isunordered", "vint32n", ["vfloatn", "vfloatn"], invoke_name="Unordered"),
                 RelDef("isunordered", "vint64n", ["vdoublen", "vdoublen"], invoke_name="Unordered"),
                 RelDef("isunordered", "bool", ["sgenfloat", "sgenfloat"], invoke_name="Unordered"),
                 RelDef("signbit", "vint16n", ["vhalfn"], invoke_name="SignBitSet"),
                 RelDef("signbit", "vint32n", ["vfloatn"], invoke_name="SignBitSet"),
                 RelDef("signbit", "vint64n", ["vdoublen"], invoke_name="SignBitSet"),
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
                 Def("select", "vint64n", ["vint64n", "vint64n", "vint64n"]),
                 Def("select", "vuint8n", ["vuint8n", "vuint8n", "vint8n"]),
                 Def("select", "vuint16n", ["vuint16n", "vuint16n", "vint16n"]),
                 Def("select", "vuint32n", ["vuint32n", "vuint32n", "vint32n"]),
                 Def("select", "vuint64n", ["vuint64n", "vuint64n", "vint64n"]),
                 Def("select", "vfloatn", ["vfloatn", "vfloatn", "vint32n"]),
                 Def("select", "vdoublen", ["vdoublen", "vdoublen", "vint64n"]),
                 Def("select", "vhalfn", ["vhalfn", "vhalfn", "vint16n"]),
                 Def("select", "vint8n", ["vint8n", "vint8n", "vuint8n"]),
                 Def("select", "vint16n", ["vint16n", "vint16n", "vuint16n"]),
                 Def("select", "vint32n", ["vint32n", "vint32n", "vuint32n"]),
                 Def("select", "vint64n", ["vint64n", "vint64n", "vuint64n"]),
                 Def("select", "vuint8n", ["vuint8n", "vuint8n", "vuint8n"]),
                 Def("select", "vuint16n", ["vuint16n", "vuint16n", "vuint16n"]),
                 Def("select", "vuint32n", ["vuint32n", "vuint32n", "vuint32n"]),
                 Def("select", "vuint64n", ["vuint64n", "vuint64n", "vuint64n"]),
                 Def("select", "vfloatn", ["vfloatn", "vfloatn", "vuint32n"]),
                 Def("select", "vdoublen", ["vdoublen", "vdoublen", "vuint64n"]),
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

def lookup_geninteger(mappings):
  if "geninteger" in mappings:
    return mappings["geninteger"]
  elif "igeninteger" in mappings:
    return mappings["igeninteger"]
  elif "ugeninteger" in mappings:
    return mappings["ugeninteger"]
  elif "vigeninteger" in mappings:
    return mappings["vigeninteger"]
  elif "vugeninteger" in mappings:
    return mappings["vugeninteger"]
  raise ValueError("No valid element type found.")

def select_from_mapping(mappings, arg_type):
  mapping = mappings[arg_type]
  # In some cases we may need to limit definitions to smaller than geninteger so
  # check for the different possible ones.
  if isinstance(mapping, ElementType):
    return InstantiatedElementType(lookup_geninteger(mappings))
  if isinstance(mapping, UnsignedType):
    return InstantiatedUnsignedType(lookup_geninteger(mappings))
  return mapping

def type_combinations(return_type, arg_types):
  unique_types = list(dict.fromkeys(arg_types + [return_type]))
  unique_type_lists = [builtin_types[unique_type] for unique_type in unique_types]
  combinations = list(itertools.product(*unique_type_lists))
  result = []
  for combination in combinations:
    mappings = dict(zip(unique_types, combination))
    mapped_return_type = select_from_mapping(mappings, return_type)
    mapped_arg_types = [select_from_mapping(mappings, arg_type) for arg_type in arg_types]
    result.append((mapped_return_type, mapped_arg_types))
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

def get_template_args(return_type, arg_types):
  has_vector = False
  has_multi_ptr = False
  for t in ([return_type] + arg_types):
    if isinstance(t, Vec):
      has_vector = True
    if isinstance(t, MultiPtr):
      has_multi_ptr = True

  result = []
  if has_vector:
    result.append("int N")
  if has_multi_ptr:
    result.append("access::address_space Space")
    result.append("access::decorated IsDecorated")
  return result

def get_func_prefix(return_type, arg_types):
  template_args = get_template_args(return_type, arg_types)
  if template_args:
    return "template <%s>" % (', '.join(template_args)) 
  return "inline"

def generate_builtin(builtin, return_type, arg_types):
  func_prefix = get_func_prefix(return_type, arg_types)
  arg_names = ["a%i" % i for i in range(len(arg_types))]
  func_args = ', '.join(["%s %s" % arg for arg in zip(arg_types, arg_names)])
  invoke = builtin.get_invoke(return_type, arg_types, arg_names)
  return f"""
{func_prefix} {return_type} {builtin.name}({func_args}) __NOEXC {{
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
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

namespace detail {
#ifdef __FAST_MATH__
  template <typename T> struct use_fast_math : is_genfloatf<T> {};
#else
  template <typename> struct use_fast_math : std::false_type {};
#endif
  template <typename T> static constexpr bool use_fast_math_v = use_fast_math<T>::value;

  // sycl::select(sgentype a, sgentype b, bool c) calls OpenCL built-in
  // select(sgentype a, sgentype b, igentype c). This type trait makes the
  // proper conversion for argument c from bool to igentype, based on sgentype
  // == T.
  template <typename T>
  using get_select_opencl_builtin_c_arg_type = typename std::conditional_t<
      sizeof(T) == 1, char,
      std::conditional_t<
          sizeof(T) == 2, short,
          std::conditional_t<
              (detail::is_contained<
                   T, detail::type_list<long, unsigned long>>::value &&
               (sizeof(T) == 4 || sizeof(T) == 8)),
              long, // long and ulong are 32-bit on
                    // Windows and 64-bit on Linux
              std::conditional_t<
                  sizeof(T) == 4, int,
                  std::conditional_t<sizeof(T) == 8, long long, void>>>>>;
} // namespace detail
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
