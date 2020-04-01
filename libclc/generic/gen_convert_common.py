# This file contains common variables and helper functions used by the
# `gen_convert.py` in both the libclc and libspirv libraries.

types = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float', 'double']
int_types = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']
unsigned_types = ['uchar', 'ushort', 'uint', 'ulong']
signed_types = ['char', 'short', 'int', 'long']
float_types = ['float', 'double']
int64_types = ['long', 'ulong']
float64_types = ['double']
vector_sizes = ['', '2', '3', '4', '8', '16']
half_sizes = {'2': '', '4': '2', '8': '4', '16': '8'}

saturation = ['','_sat']
rounding_modes = ['_rtz','_rte','_rtp','_rtn']
float_prefix = {'float':'FLT_', 'double':'DBL_'}
float_suffix = {'float':'f', 'double':''}

bool_type = {'char'  : 'char',
             'uchar' : 'char',
             'short' : 'short',
             'ushort': 'short',
             'int'   : 'int',
             'uint'  : 'int',
             'long'  : 'long',
             'ulong' : 'long',
             'float'  : 'int',
             'double' : 'long'}

unsigned_type = {'char'  : 'uchar',
                 'uchar' : 'uchar',
                 'short' : 'ushort',
                 'ushort': 'ushort',
                 'int'   : 'uint',
                 'uint'  : 'uint',
                 'long'  : 'ulong',
                 'ulong' : 'ulong'}

sizeof_type = {'char'  : 1, 'uchar'  : 1,
               'short' : 2, 'ushort' : 2,
               'int'   : 4, 'uint'   : 4,
               'long'  : 8, 'ulong'  : 8,
               'float' : 4, 'double' : 8}

limit_max = {'char'  : 'CHAR_MAX',
             'uchar' : 'UCHAR_MAX',
             'short' : 'SHRT_MAX',
             'ushort': 'USHRT_MAX',
             'int'   : 'INT_MAX',
             'uint'  : 'UINT_MAX',
             'long'  : 'LONG_MAX',
             'ulong' : 'ULONG_MAX'}

limit_min = {'char'  : 'CHAR_MIN',
             'uchar' : '0',
             'short' : 'SHRT_MIN',
             'ushort': '0',
             'int'   : 'INT_MIN',
             'uint'  : '0',
             'long'  : 'LONG_MIN',
             'ulong' : '0'}


def conditional_guard(src, dst):
  """
  This function will optionally print a header guard for `cl_khr_fp64` if a 64-bit type is used
  as the source or destination and return a bool that indicates whether this guard will need
  closed after the calling function has finished printing functions that use the 64-bit
  source/destination type.
  """
  int64_count = 0
  float64_count = 0
  if src in int64_types:
    int64_count = int64_count +1
  elif src in float64_types:
    float64_count = float64_count + 1
  if dst in int64_types:
    int64_count = int64_count +1
  elif dst in float64_types:
    float64_count = float64_count + 1
  if float64_count > 0:
    #In embedded profile, if cl_khr_fp64 is supported cles_khr_int64 has to be
    print("#ifdef cl_khr_fp64")
    return True
  elif int64_count > 0:
    print("#if defined cles_khr_int64 || !defined(__EMBEDDED_PROFILE__)")
    return True
  return False



def spirv_fn_name(src, dst, size='', mode='', sat=''):
  """
  This helper function returns the correct SPIR-V function name for a given source and destination
  type, with optional size, mode and saturation arguments.
  """
  is_src_float = src in float_types
  is_src_unsigned = src in unsigned_types
  is_src_signed = src in signed_types
  is_dst_float = dst in float_types
  is_dst_unsigned = dst in unsigned_types
  is_dst_signed = dst in signed_types
  is_sat = sat != ''

  if is_src_unsigned and is_dst_signed and is_sat:
    return '__spirv_SatConvertUToS_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_signed and is_dst_unsigned and is_sat:
    return '__spirv_SatConvertSToU_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_float and is_dst_signed:
    return '__spirv_ConvertFToS_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_float and is_dst_unsigned:
    return '__spirv_ConvertFToU_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_signed and is_dst_float:
    return '__spirv_ConvertSToF_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_unsigned and is_dst_float:
    return '__spirv_ConvertUToF_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_float and is_dst_float:
    return '__spirv_FConvert_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_unsigned and is_dst_unsigned:
    return '__spirv_UConvert_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  elif is_src_signed and is_dst_signed:
    return '__spirv_SConvert_R{DST}{N}{MODE}'.format(DST=dst, N=size, MODE=mode)
  else:
    return None
