//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                  void *HostPtr, RT::PiEvent &OutEventToWait) {

  assert(!(InitFromUserData && HostPtr) &&
          "Cannot init from user data and reuse host ptr provided "
          "simultaneously");

  void *UserPtr = InitFromUserData ? BaseT::getUserPtr() : HostPtr;

  assert(!(nullptr == UserPtr && BaseT::useHostPtr() && Context->is_host()) &&
          "Internal error. Allocating memory on the host "
          "while having use_host_ptr property");

  return MemoryManager::allocateMemBuffer(
      std::move(Context), this, UserPtr, BaseT::MHostPtrReadOnly,
      BaseT::getSize(), BaseT::MInteropEvent, BaseT::MInteropContext,
      OutEventToWait);
}
} // namespace detail
} // namespace sycl
} // namespace cl
