// TODO: Enable half precision when sin/cos is implemented
#if __CLC_FPSIZE > 16
#define __CLC_DECLARE_SINCOS(ADDRSPACE, TYPE) \
  _CLC_OVERLOAD _CLC_DEF TYPE sincos (TYPE x, ADDRSPACE TYPE * cosval) { \
    return __spirv_ocl_sincos(x, cosval); \
  }

__CLC_DECLARE_SINCOS(global, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(local, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(private, __CLC_GENTYPE)
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
__CLC_DECLARE_SINCOS(generic, __CLC_GENTYPE)
#endif

#undef __CLC_DECLARE_SINCOS
#endif
