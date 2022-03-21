//==-------------------- host_pipe_map.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/experimental/host_pipes.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

template <class _name, typename _dataT, typename PropertyList>
_dataT host_pipe<_name, _dataT, PropertyList>::read(queue &q,
                                                    memory_order order) {
  const device Dev = q.get_device();
  bool IsReadPipeSupported =
      Dev.has_extension("cl_intel_program_scope_host_pipe");
  if (!IsReadPipeSupported) {
    return &_dataT();
  }
  // TODO: Get pipe name from template, get host pointer by quering the host
  // pipe registration / host pipe mapping
  _dataT data;
  const std::string pipe_name = std::to_string(ID);
  size_t size = 4;
  // event e = q.read_host_pipe(name, (void*)pipe_ptr, (size_t)size, false);
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, (void *)(&data), (size_t)size, false,
                             false /* write */);
  });
  e.wait();
  return data;
}

template <class _name, typename _dataT, typename PropertyList>
void host_pipe<_name, _dataT, PropertyList>::write(queue &q, const _dataT &data,
                                                   memory_order order) {
  const device Dev = q.get_device();
  bool IsReadPipeSupported =
      Dev.has_extension("cl_intel_program_scope_host_pipe");
  if (!IsReadPipeSupported) {
    return;
  }
  // TODO: is this the correct logical name of the pipe?
  const std::string pipe_name = std::to_string(ID);
  const void *data_ptr = &data;
  size_t size = 4;
  // event e = q.write_host_pipe(name, (void*)pipe_ptr, (size_t)size, false);
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
