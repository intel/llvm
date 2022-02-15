#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
  namespace sycl {
  namespace detail {
  namespace device_global_map {

  void add(void *DeviceGlobalPtr, const char *UniqueId);

  } // namespace device_global_map
  } // namespace detail
  } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
