//==-------------------- host_pipe.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/experimental/host_pipes.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace experimental {

template <class _name, class _dataT, typename _propertiesT>
_dataT
host_pipe<_name, _dataT, _propertiesT,
          std::enable_if_t<sycl::ext::oneapi::experimental::is_property_list_v<
              _propertiesT>>>::read(queue &q, memory_order order) {
  const device Dev = q.get_device();
  bool IsReadPipeSupported =
      Dev.has_extension("cl_intel_program_scope_host_pipe");
  if (!IsReadPipeSupported) {
    return &_dataT();
  }
  // TODO: get pipe name from the pipe registration
  _dataT data;
  const std::string pipe_name = "pipename";
  size_t size = 4;
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, (void *)(&data), (size_t)size, false,
                             true /* read */);
  });
  e.wait();
  return data;
}

template <class _name, class _dataT, typename _propertiesT>
void host_pipe<
    _name, _dataT, _propertiesT,
    std::enable_if_t<sycl::ext::oneapi::experimental::is_property_list_v<
        _propertiesT>>>::write(queue &q, const _dataT &data,
                               memory_order order) {
  const device Dev = q.get_device();
  bool IsReadPipeSupported =
      Dev.has_extension("cl_intel_program_scope_host_pipe");
  if (!IsReadPipeSupported) {
    return;
  }
  // TODO: get pipe name from the pipe registration
  const std::string pipe_name = "pipename";
  const void *data_ptr = &data;
  size_t size = 4;
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, (void *)data_ptr, (size_t)size, false,
                             false /* write */);
  });
  e.wait();
}

// TODO: implement non blocking version

} // namespace experimental
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl