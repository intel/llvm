# This file contains common variables and helper functions used by the
# `gen_convert.py` in both the libclc and libspirv libraries.

types = ['char', 'schar', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'half', 'float', 'double']
int_types = ['char', 'schar', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']
unsigned_types = ['uchar', 'ushort', 'uint', 'ulong']
signed_types = ['char', 'schar', 'short', 'int', 'long']
float_types = ['half', 'float', 'double']
int64_types = ['long', 'ulong']
float64_types = ['double']
float16_types = ['half']
vector_sizes = ['', '2', '3', '4', '8', '16']
half_sizes = [('2', ''), ('4', '2'), ('8', '4'), ('16', '8')]

saturation = ['','_sat']
rounding_modes = ['_rtz','_rte','_rtp','_rtn']
float_prefix = {'float':'FLT_', 'double':'DBL_'}
float_suffix = {'float':'f', 'double':''}

bool_type = {'char'   : 'char',
             'schar'  : 'schar',
             'uchar'  : 'schar',
             'short'  : 'short',
             'ushort' : 'short',
             'int'    : 'int',
             'uint'   : 'int',
             'long'   : 'long',
             'ulong'  : 'long',
             'half'   : 'short',
             'float'  : 'int',
             'double' : 'long'}

unsigned_type = {'char'  : 'uchar',
                 'schar' : 'uchar',
                 'uchar' : 'uchar',
                 'short' : 'ushort',
                 'ushort': 'ushort',
                 'int'   : 'uint',
                 'uint'  : 'uint',
                 'long'  : 'ulong',
                 'ulong' : 'ulong'}

sizeof_type = {'char'  : 1, 'schar'  : 1, 'uchar'  : 1,
               'short' : 2, 'ushort' : 2,
               'int'   : 4, 'uint'   : 4,
               'long'  : 8, 'ulong'  : 8,
               'half'  : 2, 'float'  : 4,
               'double': 8}

limit_max = {'char'  : 'CHAR_MAX',
             'schar' : 'CHAR_MAX',
             'uchar' : 'UCHAR_MAX',
             'short' : 'SHRT_MAX',
             'ushort': 'USHRT_MAX',
             'int'   : 'INT_MAX',
             'uint'  : 'UINT_MAX',
             'long'  : 'LONG_MAX',
             'ulong' : 'ULONG_MAX',
             'half'  : '0x1.ffcp+15'
             }

limit_min = {'char'  : 'CHAR_MIN',
             'schar' : 'CHAR_MIN',
             'uchar' : '0',
             'short' : 'SHRT_MIN',
             'ushort': '0',
             'int'   : 'INT_MIN',
             'uint'  : '0',
             'long'  : 'LONG_MIN',
             'ulong' : '0',
             'half'  : '-0x1.ffcp+15'
             }



def conditional_guard(src, dst):
  """
  This function will optionally print a header guard for `cl_khr_fp64` if a 64-bit type is used
  as the source or destination and return a bool that indicates whether this guard will need
  closed after the calling function has finished printing functions that use the 64-bit
  source/destination type.
  """
  int64_count = 0
  float64_count = 0
  float16_count = 0
  if src in int64_types or dst in int64_types:
    int64_count = 1
  if src in float64_types or dst in float64_types:
    float64_count = 1
  if src in float16_types or dst in float16_types:
    float16_count = 1
  if float16_count > 0:
    print("#ifdef cl_khr_fp16")
  if float64_count > 0:
    #In embedded profile, if cl_khr_fp64 is supported cles_khr_int64 has to be
    print("#ifdef cl_khr_fp64")
    return 1 + float16_count
  elif int64_count > 0:
    print("#if defined cles_khr_int64 || !defined(__EMBEDDED_PROFILE__)")
    return 1 + float16_count
  return float16_count

def close_conditional_guard(close_conditional):
  """
  This function will close conditional guard opened by conditional_guard.
  """
  for _ in range(close_conditional):
    print("#endif")

def clc_core_fn_name(dst, size='', mode='', sat=''):
  """
  This helper function returns the correct clc core conversion function name
  for a given source and destination type, with optional size, mode
  and saturation arguments.
  """
  return "__clc_convert_{DST}{N}{SAT}{MODE}".format(DST=dst, N=size, SAT=sat, MODE=mode)
