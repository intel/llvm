// This does not exist for fp16
#if __CLC_FPSIZE > 16
#ifdef __CLC_VECSIZE

#if __CLC_VECSIZE == 3
#  define __CLC_OFFSET 4
#else
#  define __CLC_OFFSET __CLC_VECSIZE
#endif

  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __global);
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __generic);
#endif

#undef __CLC_OFFSET
#else
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __private);
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __local);
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __global);
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __generic);
#endif
#endif
#endif
