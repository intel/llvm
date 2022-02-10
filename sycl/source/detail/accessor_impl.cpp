//==---------------- accessor_impl.cpp - SYCL standard source file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/buffer_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

AccessorImplHost::~AccessorImplHost() {
  try {
    if (MBlockedCmd)
      detail::Scheduler::getInstance().releaseHostAccessor(this);
  } catch (...) {
  }
}

void AccessorImplHost::resize(size_t GlobalSize) {
  if (GlobalSize != 1) {
    auto Bufi = static_cast<detail::buffer_impl *>(MSYCLMemObj);
    MMemoryRange[0] *= GlobalSize;
    MAccessRange[0] *= GlobalSize;
    Bufi->resize(MMemoryRange[0] * MElemSize);
  }
}

void addHostAccessorAndWait(Requirement *Req) {
  detail::EventImplPtr Event =
      detail::Scheduler::getInstance().addHostAccessor(Req);
  Event->wait(Event);
}

void constructorNotification(void *BufferObj, void *AccessorObj,
                             cl::sycl::access::target Target,
                             cl::sycl::access::mode Mode,
                             const detail::code_location &CodeLoc) {
  XPTIRegistry::bufferAccessorNotification(
      BufferObj, AccessorObj, (uint32_t)Target, (uint32_t)Mode, CodeLoc);
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
