#define __CLC_CONCAT(a, b, c) a ## b ## c
#define __CLC_XCONCAT(a, b, c) __CLC_CONCAT(a, b, c)

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    global __CLC_GENTYPE *dst,
    const local __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {

  return __spirv_GroupAsyncCopy(Workgroup, dst, src, num_gentypes, stride,
                                event);
}

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    local __CLC_GENTYPE *dst,
    const global __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t stride,
    event_t event) {
  return __spirv_GroupAsyncCopy(Workgroup, dst, src, num_gentypes, stride,
                                event);
}

#undef __CLC_XCONCAT
#undef __CLC_CONCAT
