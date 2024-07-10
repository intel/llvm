//==------------ accessor.cpp - SYCL standard source file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/accessor.hpp>
#include <sycl/accessor_image.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
device getDeviceFromHandler(handler &cgh) {
  assert((cgh.MQueue || cgh.MGraph) &&
         "One of MQueue or MGraph should be nonnull!");
  if (cgh.MQueue)
    return cgh.MQueue->get_device();

  return cgh.MGraph->getDevice();
}

AccessorBaseHost::AccessorBaseHost(id<3> Offset, range<3> AccessRange,
                                   range<3> MemoryRange,
                                   access::mode AccessMode, void *SYCLMemObject,
                                   int Dims, int ElemSize, size_t OffsetInBytes,
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
                                   size_t OffsetInBytes, bool IsSubBuffer,
                                   const property_list &PropertyList) {
  impl = std::shared_ptr<AccessorImplHost>(
      new AccessorImplHost(Offset, AccessRange, MemoryRange, AccessMode,
                           (detail::SYCLMemObjI *)SYCLMemObject, Dims, ElemSize,
                           IsPlaceH, OffsetInBytes, IsSubBuffer, PropertyList));
}

id<3> &AccessorBaseHost::getOffset() { return impl->MOffset; }
range<3> &AccessorBaseHost::getAccessRange() { return impl->MAccessRange; }
range<3> &AccessorBaseHost::getMemoryRange() { return impl->MMemoryRange; }
void *AccessorBaseHost::getPtr() noexcept { return impl->MData; }

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
void *AccessorBaseHost::getPtr() const noexcept {
  return const_cast<void *>(impl->MData);
}

void *AccessorBaseHost::getMemoryObject() const { return impl->MSYCLMemObj; }

bool AccessorBaseHost::isPlaceholder() const { return impl->MIsPlaceH; }

bool AccessorBaseHost::isMemoryObjectUsedByGraph() const {
  return static_cast<detail::SYCLMemObjT *>(impl->MSYCLMemObj)->isUsedInGraph();
}

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

UnsampledImageAccessorBaseHost::UnsampledImageAccessorBaseHost(
    sycl::range<3> Size, access_mode AccessMode, void *SYCLMemObject, int Dims,
    int ElemSize, id<3> Pitch, image_channel_type ChannelType,
    image_channel_order ChannelOrder, const property_list &PropertyList) {
  impl = std::make_shared<UnsampledImageAccessorImplHost>(
      Size, AccessMode, (detail::SYCLMemObjI *)SYCLMemObject, Dims, ElemSize,
      Pitch, ChannelType, ChannelOrder, PropertyList);
}
const sycl::range<3> &UnsampledImageAccessorBaseHost::getSize() const {
  return impl->MAccessRange;
}
const property_list &UnsampledImageAccessorBaseHost::getPropList() const {
  return impl->MPropertyList;
}
void *UnsampledImageAccessorBaseHost::getMemoryObject() const {
  return impl->MSYCLMemObj;
}
detail::AccHostDataT &UnsampledImageAccessorBaseHost::getAccData() {
  return impl->MAccData;
}
void *UnsampledImageAccessorBaseHost::getPtr() { return impl->MData; }
void *UnsampledImageAccessorBaseHost::getPtr() const {
  return const_cast<void *>(impl->MData);
}
int UnsampledImageAccessorBaseHost::getNumOfDims() const { return impl->MDims; }
int UnsampledImageAccessorBaseHost::getElementSize() const {
  return impl->MElemSize;
}
id<3> UnsampledImageAccessorBaseHost::getPitch() const { return impl->MPitch; }
image_channel_type UnsampledImageAccessorBaseHost::getChannelType() const {
  return impl->MChannelType;
}
image_channel_order UnsampledImageAccessorBaseHost::getChannelOrder() const {
  return impl->MChannelOrder;
}

SampledImageAccessorBaseHost::SampledImageAccessorBaseHost(
    sycl::range<3> Size, void *SYCLMemObject, int Dims, int ElemSize,
    id<3> Pitch, image_channel_type ChannelType,
    image_channel_order ChannelOrder, image_sampler Sampler,
    const property_list &PropertyList) {
  impl = std::make_shared<SampledImageAccessorImplHost>(
      Size, (detail::SYCLMemObjI *)SYCLMemObject, Dims, ElemSize, Pitch,
      ChannelType, ChannelOrder, Sampler, PropertyList);
}
const sycl::range<3> &SampledImageAccessorBaseHost::getSize() const {
  return impl->MAccessRange;
}
const property_list &SampledImageAccessorBaseHost::getPropList() const {
  return impl->MPropertyList;
}
void *SampledImageAccessorBaseHost::getMemoryObject() const {
  return impl->MSYCLMemObj;
}
detail::AccHostDataT &SampledImageAccessorBaseHost::getAccData() {
  return impl->MAccData;
}
void *SampledImageAccessorBaseHost::getPtr() { return impl->MData; }
void *SampledImageAccessorBaseHost::getPtr() const {
  return const_cast<void *>(impl->MData);
}
int SampledImageAccessorBaseHost::getNumOfDims() const { return impl->MDims; }
int SampledImageAccessorBaseHost::getElementSize() const {
  return impl->MElemSize;
}
id<3> SampledImageAccessorBaseHost::getPitch() const { return impl->MPitch; }
image_channel_type SampledImageAccessorBaseHost::getChannelType() const {
  return impl->MChannelType;
}
image_channel_order SampledImageAccessorBaseHost::getChannelOrder() const {
  return impl->MChannelOrder;
}
image_sampler SampledImageAccessorBaseHost::getSampler() const {
  return impl->MSampler;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
