//==-------------------- pipes.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

__SYCL_EXPORT std::string pipe_base::get_pipe_name(const void *HostPipePtr) {
  return sycl::_V1::detail::ProgramManager::getInstance()
      .getHostPipeEntry(HostPipePtr)
      ->MUniqueId;
}

__SYCL_EXPORT bool pipe_base::wait_non_blocking(const event &E) {
  bool Success = false;
  std::shared_ptr<sycl::detail::event_impl> EImpl =
      sycl::detail::getSyclObjImpl(E);
  EImpl->wait(EImpl, &Success);
  return Success;
}

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl
