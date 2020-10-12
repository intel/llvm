#include <clc/clc.h>
#include <spirv/spirv.h>


_CLC_DEF _CLC_OVERLOAD size_t get_global_id(uint dim) {
switch (dim) {
    case 0:  return __spirv_GlobalInvocationId_x();
    case 1:  return __spirv_GlobalInvocationId_y();
    case 2:  return __spirv_GlobalInvocationId_z();
    default: return 0;
  }
}
