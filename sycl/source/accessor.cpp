//==------------ accessor.cpp - SYCL standard source file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/accessor.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
device getDeviceFromHandler(handler &CommandGroupHandlerRef) {
  return CommandGroupHandlerRef.MQueue->get_device();
}

AccessorBaseHost::AccessorBaseHost(id<3> Offset, range<3> AccessRange,
                                   range<3> MemoryRange,
                                   access::mode AccessMode, void *SYCLMemObject,
                                   int Dims, int ElemSize, int OffsetInBytes,
                                   bool IsSubBuffer,
                                   const property_list &PropertyList) {
  impl = std::shared_ptr<AccessorImplHost>(
      new AccessorImplHost(Offset, AccessRange, MemoryRange, AccessMode,
                           (detail::SYCLMemObjI *)SYCLMemObject, Dims, ElemSize,
                           false, OffsetInBytes, IsSubBuffer, PropertyList));
}

AccessorBaseHost::AccessorBaseHost(id<3> Offset, range<3> AccessRange,
                                   range<3> MemoryRange,
                                   access::mode AccessMode, void *SYCLMemObject,
                                   int Dims, int ElemSize, bool IsPlaceH,
                                   int OffsetInBytes, bool IsSubBuffer,
                                   const property_list &PropertyList) {
  impl = std::shared_ptr<AccessorImplHost>(
      new AccessorImplHost(Offset, AccessRange, MemoryRange, AccessMode,
                           (detail::SYCLMemObjI *)SYCLMemObject, Dims, ElemSize,
                           IsPlaceH, OffsetInBytes, IsSubBuffer, PropertyList));
}

id<3> &AccessorBaseHost::getOffset() { return impl->MOffset; }
range<3> &AccessorBaseHost::getAccessRange() { return impl->MAccessRange; }
range<3> &AccessorBaseHost::getMemoryRange() { return impl->MMemoryRange; }
void *AccessorBaseHost::getPtr() { return impl->MData; }

detail::AccHostDataT &AccessorBaseHost::getAccData() { return impl->MAccData; }

const property_list &AccessorBaseHost::getPropList() const {
  return impl->MPropertyList;
}

unsigned int AccessorBaseHost::getElemSize() const { return impl->MElemSize; }

const id<3> &AccessorBaseHost::getOffset() const { return impl->MOffset; }
const range<3> &AccessorBaseHost::getAccessRange() const {
  return impl->MAccessRange;
}
const range<3> &AccessorBaseHost::getMemoryRange() const {
  return impl->MMemoryRange;
}
void *AccessorBaseHost::getPtr() const {
  return const_cast<void *>(impl->MData);
}

void *AccessorBaseHost::getMemoryObject() const { return impl->MSYCLMemObj; }

bool AccessorBaseHost::isPlaceholder() const { return impl->MIsPlaceH; }

LocalAccessorBaseHost::LocalAccessorBaseHost(
    sycl::range<3> Size, int Dims, int ElemSize,
    const property_list &PropertyList) {
  impl = std::shared_ptr<LocalAccessorImplHost>(
      new LocalAccessorImplHost(Size, Dims, ElemSize, PropertyList));
}
sycl::range<3> &LocalAccessorBaseHost::getSize() { return impl->MSize; }
const sycl::range<3> &LocalAccessorBaseHost::getSize() const {
  return impl->MSize;
}
void *LocalAccessorBaseHost::getPtr() {
  // Const cast this in order to call the const getPtr.
  return const_cast<const LocalAccessorBaseHost *>(this)->getPtr();
}
void *LocalAccessorBaseHost::getPtr() const {
  char *ptr = impl->MMem.data();

  // Align the pointer to MElemSize.
  size_t val = reinterpret_cast<size_t>(ptr);
  if (val % impl->MElemSize != 0) {
    ptr += impl->MElemSize - val % impl->MElemSize;
  }

  return ptr;
}
const property_list &LocalAccessorBaseHost::getPropList() const {
  return impl->MPropertyList;
}

int LocalAccessorBaseHost::getNumOfDims() { return impl->MDims; }
int LocalAccessorBaseHost::getElementSize() { return impl->MElemSize; }

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
