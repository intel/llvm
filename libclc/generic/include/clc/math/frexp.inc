_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE frexp(__CLC_GENTYPE x, global __CLC_INTN *iptr);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE frexp(__CLC_GENTYPE x, local __CLC_INTN *iptr);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE frexp(__CLC_GENTYPE x, private __CLC_INTN *iptr);
#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE frexp(__CLC_GENTYPE x, generic __CLC_INTN *iptr);
#endif
