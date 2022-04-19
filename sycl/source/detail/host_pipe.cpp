//==-------------------- host_pipe.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/experimental/host_pipes.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <detail/program_manager/program_manager.hpp>

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
  const void *HostPipePtr = &__pipe;
  detail::HostPipeMapEntry hostPipeEntry =
      detail::ProgramManager::getInstance().getHostPipeEntry(HostPipePtr);
  const std::string pipe_name = hostPipeEntry.MUniqueId;
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, (void *)(&data), sizeof(_dataT), false,
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
  const void *HostPipePtr = &__pipe;
  detail::HostPipeMapEntry hostPipeEntry =
      detail::ProgramManager::getInstance().getHostPipeEntry(HostPipePtr);
  const std::string pipe_name = hostPipeEntry.MUniqueId;
  const void *data_ptr = &data;
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, (void *)data_ptr, sizeof(_dataT), false,
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