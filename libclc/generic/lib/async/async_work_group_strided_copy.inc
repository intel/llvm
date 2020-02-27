#define __CLC_CONCAT(a, b, c) a ## b ## c
#define __CLC_XCONCAT(a, b, c) __CLC_CONCAT(a, b, c)

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    global __CLC_GENTYPE *dst,
    const local __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {

  return __CLC_XCONCAT(_Z22__spirv_GroupAsyncCopyI, __CLC_GENTYPE_MANGLED, E9ocl_eventN5__spv5ScopeEPU3AS1T_PU3AS3S3_mmS0_)(Workgroup, dst, src, num_gentypes, stride, event);
}

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    local __CLC_GENTYPE *dst,
    const global __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {
  return __CLC_XCONCAT(_Z22__spirv_GroupAsyncCopyI, __CLC_GENTYPE_MANGLED, E9ocl_eventN5__spv5ScopeEPU3AS3T_PU3AS1S3_mmS0_)(Workgroup, dst, src, num_gentypes, stride, event);
}

#undef __CLC_XCONCAT
#undef __CLC_CONCAT
