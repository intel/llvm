#if __CLC_FPSIZE == 32

#ifdef __CLC_VECSIZE

#if __CLC_VECSIZE == 3
#  define __CLC_OFFSET 4
#else
#  define __CLC_OFFSET __CLC_VECSIZE
#endif

  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __global);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __constant);
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __generic);
#endif

#undef __CLC_OFFSET
#else
  FUNC(, 1, 1, __CLC_GENTYPE, __private);
  FUNC(, 1, 1, __CLC_GENTYPE, __local);
  FUNC(, 1, 1, __CLC_GENTYPE, __global);
  FUNC(, 1, 1, __CLC_GENTYPE, __constant);
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
  FUNC(, 1, 1, __CLC_GENTYPE, __generic);
#endif
#endif
#endif
