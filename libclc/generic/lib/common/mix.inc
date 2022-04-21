_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_GENTYPE a) {
  return __spirv_ocl_mix(x, y, a);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_SCALAR_GENTYPE a) {
    return __spirv_ocl_mix(x, y, a);
}
#endif
