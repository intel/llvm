//==---------------- accessor_impl.cpp - SYCL standard source file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/accessor_impl.hpp>
#include <detail/accessor_impl_host.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

AccessorImplHost::AccessorImplHost(id<3> Offset, range<3> AccessRange,
                                   range<3> MemoryRange,
                                   access::mode AccessMode,
                                   detail::SYCLMemObjI *SYCLMemObject, int Dims,
                                   int ElemSize, int OffsetInBytes,
                                   bool IsSubBuffer)
    : MOffset(Offset), MAccessRange(AccessRange), MMemoryRange(MemoryRange),
      MAccessMode(AccessMode), MSYCLMemObj(SYCLMemObject), MDims(Dims),
      MElemSize(ElemSize), MOffsetInBytes(OffsetInBytes),
      MIsSubBuffer(IsSubBuffer) {}

AccessorImplHost::AccessorImplHost(const AccessorImplHost &Other)
    : MOffset(Other.MOffset), MAccessRange(Other.MAccessRange),
      MMemoryRange(Other.MMemoryRange), MAccessMode(Other.MAccessMode),
      MSYCLMemObj(Other.MSYCLMemObj), MDims(Other.MDims),
      MElemSize(Other.MElemSize), MOffsetInBytes(Other.MOffsetInBytes),
      MIsSubBuffer(Other.MIsSubBuffer) {}

AccessorImplHost::~AccessorImplHost() {
  if (MBlockedCmd)
    detail::Scheduler::getInstance().releaseHostAccessor(this);
}

AccessorBaseHost::AccessorBaseHost(id<3> Offset, range<3> AccessRange,
                                   range<3> MemoryRange,
                                   access::mode AccessMode,
                                   detail::SYCLMemObjI *SYCLMemObject, int Dims,
                                   int ElemSize, int OffsetInBytes,
                                   bool IsSubBuffer)
    : MCachedOffset(Offset), MCachedAccessRange(AccessRange),
      MCachedMemoryRange(MemoryRange), MCachedElemSize(ElemSize) {
  impl = shared_ptr_class<AccessorImplHost>(new AccessorImplHost(
      Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject, Dims,
      ElemSize, OffsetInBytes, IsSubBuffer));
}

void *AccessorBaseHost::getPtr() const { return impl->MData; }

LocalAccessorImplHost::LocalAccessorImplHost(sycl::range<3> Size, int Dims,
                                             int ElemSize)
    : MSize(Size), MDims(Dims), MElemSize(ElemSize),
      MMem(Size[0] * Size[1] * Size[2] * ElemSize) {}

void LocalAccessorImplHost::resize(size_t LocalSize, size_t GlobalSize) {
  if (GlobalSize != 1 && LocalSize != 1) {
    // If local size is not specified then work group size is chosen by
    // runtime. That is why try to allocate based on max work group size or
    // global size. In the worst case allocate 80% of local memory.
    size_t MinEstWGSize = LocalSize ? LocalSize : GlobalSize;
    MinEstWGSize = MinEstWGSize > MaxWGSize ? MaxWGSize : MinEstWGSize;
    size_t NewSize = MinEstWGSize * MSize[0];
    MSize[0] =
        NewSize > 8 * LocalMemSize / 10 ? 8 * LocalMemSize / 10 : NewSize;
    MMem.resize(NewSize * MElemSize);
  }
}

LocalAccessorBaseHost::LocalAccessorBaseHost(sycl::range<3> Size, int Dims,
                                             int ElemSize) {
  impl = shared_ptr_class<LocalAccessorImplHost>(
      new LocalAccessorImplHost(Size, Dims, ElemSize));
}

void addHostAccessorAndWait(Requirement *Req) {
  detail::EventImplPtr Event =
      detail::Scheduler::getInstance().addHostAccessor(Req);
  Event->wait(Event);
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
