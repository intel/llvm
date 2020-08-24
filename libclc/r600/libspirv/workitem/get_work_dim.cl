#include <spirv/spirv.h>

<<<<<<< HEAD:libclc/r600/libspirv/workitem/get_work_dim.cl
_CLC_DEF _CLC_OVERLOAD uint __spirv_WorkDim(void)
{
	__attribute__((address_space(7))) uint * ptr =
		(__attribute__((address_space(7))) uint *)
		__builtin_r600_implicitarg_ptr();
	return ptr[0];
=======
_CLC_DEF _CLC_OVERLOAD uint get_work_dim(void) {
  __attribute__((address_space(7))) uint *ptr =
      (__attribute__((address_space(7)))
       uint *)__builtin_r600_implicitarg_ptr();
  return ptr[0];
>>>>>>> 3d21fa56f5f5afbbf16b35b199480af71e1189a3:libclc/r600/lib/workitem/get_work_dim.cl
}
