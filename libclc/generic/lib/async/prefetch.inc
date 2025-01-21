_CLC_OVERLOAD _CLC_DEF void prefetch(const global __CLC_GENTYPE *p,
                                     size_t num_gentypes) {
  __spirv_ocl_prefetch(p, num_gentypes);
}
