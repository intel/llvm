//==---------------- accessor_impl.cpp - SYCL standard source file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/accessor_impl.hpp>
#include <detail/buffer_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

void addHostUnsampledImageAccessorAndWait(UnsampledImageAccessorImplHost *Req) {
  addHostAccessorAndWait(Req);
}

void addHostSampledImageAccessorAndWait(SampledImageAccessorImplHost *Req) {
  addHostAccessorAndWait(Req);
}

void constructorNotification(void *BufferObj, void *AccessorObj,
                             sycl::access::target Target,
                             sycl::access::mode Mode,
                             const detail::code_location &CodeLoc) {
  XPTIRegistry::bufferAccessorNotification(
      BufferObj, AccessorObj, (uint32_t)Target, (uint32_t)Mode, CodeLoc);
}

void unsampledImageConstructorNotification(void *ImageObj, void *AccessorObj,
                                           std::optional<image_target> Target,
                                           access::mode Mode, const void *Type,
                                           uint32_t ElemSize,
                                           const code_location &CodeLoc) {
  if (Target)
    XPTIRegistry::unsampledImageAccessorNotification(
        ImageObj, AccessorObj, (uint32_t)*Target, (uint32_t)Mode, Type,
        ElemSize, CodeLoc);
  else
    XPTIRegistry::unsampledImageHostAccessorNotification(
        ImageObj, AccessorObj, (uint32_t)Mode, Type, ElemSize, CodeLoc);
}

void sampledImageConstructorNotification(void *ImageObj, void *AccessorObj,
                                         std::optional<image_target> Target,
                                         const void *Type, uint32_t ElemSize,
                                         const code_location &CodeLoc) {
  if (Target)
    XPTIRegistry::sampledImageAccessorNotification(
        ImageObj, AccessorObj, (uint32_t)*Target, Type, ElemSize, CodeLoc);
  else
    XPTIRegistry::sampledImageHostAccessorNotification(ImageObj, AccessorObj,
                                                       Type, ElemSize, CodeLoc);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
