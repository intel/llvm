//==--------------------- accessor_image_base.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/image_accessor_util.hpp>
#include <sycl/id.hpp>
#include <sycl/image.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

struct AccHostDataT;

class UnsampledImageAccessorImplHost;
class SampledImageAccessorImplHost;
using UnsampledImageAccessorImplPtr =
    std::shared_ptr<UnsampledImageAccessorImplHost>;
using SampledImageAccessorImplPtr =
    std::shared_ptr<SampledImageAccessorImplHost>;

class __SYCL_EXPORT UnsampledImageAccessorBaseHost {
protected:
  UnsampledImageAccessorBaseHost(const UnsampledImageAccessorImplPtr &Impl)
      : impl{Impl} {}

public:
  UnsampledImageAccessorBaseHost(sycl::range<3> Size, access_mode AccessMode,
                                 void *SYCLMemObject, int Dims, int ElemSize,
                                 id<3> Pitch, image_channel_type ChannelType,
                                 image_channel_order ChannelOrder,
                                 const property_list &PropertyList = {});
  const sycl::range<3> &getSize() const;
  void *getMemoryObject() const;
  detail::AccHostDataT &getAccData();
  void *getPtr();
  void *getPtr() const;
  int getNumOfDims() const;
  int getElementSize() const;
  id<3> getPitch() const;
  image_channel_type getChannelType() const;
  image_channel_order getChannelOrder() const;
  const property_list &getPropList() const;

protected:
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  UnsampledImageAccessorImplPtr impl;

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getSize();
    (void)this_const->getSize();
    (void)getPtr();
    (void)this_const->getPtr();
#endif
  }

#ifndef __SYCL_DEVICE_ONLY__
  // Reads a pixel of the underlying image at the specified coordinate. It is
  // the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  DataT read(const CoordT &Coords) const noexcept {
    image_sampler Smpl{addressing_mode::none,
                       coordinate_normalization_mode::unnormalized,
                       filtering_mode::nearest};
    return imageReadSamplerHostImpl<CoordT, DataT>(
        Coords, Smpl, getSize(), getPitch(), getChannelType(),
        getChannelOrder(), getPtr(), getElementSize());
  }

  // Writes to a pixel of the underlying image at the specified coordinate. It
  // is the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  void write(const CoordT &Coords, const DataT &Color) const {
    imageWriteHostImpl(Coords, Color, getPitch(), getElementSize(),
                       getChannelType(), getChannelOrder(), getPtr());
  }
#endif
};

class __SYCL_EXPORT SampledImageAccessorBaseHost {
protected:
  SampledImageAccessorBaseHost(const SampledImageAccessorImplPtr &Impl)
      : impl{Impl} {}

public:
  SampledImageAccessorBaseHost(sycl::range<3> Size, void *SYCLMemObject,
                               int Dims, int ElemSize, id<3> Pitch,
                               image_channel_type ChannelType,
                               image_channel_order ChannelOrder,
                               image_sampler Sampler,
                               const property_list &PropertyList = {});
  const sycl::range<3> &getSize() const;
  void *getMemoryObject() const;
  detail::AccHostDataT &getAccData();
  void *getPtr();
  void *getPtr() const;
  int getNumOfDims() const;
  int getElementSize() const;
  id<3> getPitch() const;
  image_channel_type getChannelType() const;
  image_channel_order getChannelOrder() const;
  image_sampler getSampler() const;
  const property_list &getPropList() const;

protected:
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  SampledImageAccessorImplPtr impl;

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getSize();
    (void)this_const->getSize();
    (void)getPtr();
    (void)this_const->getPtr();
#endif
  }

#ifndef __SYCL_DEVICE_ONLY__
  // Reads a pixel of the underlying image at the specified coordinate. It is
  // the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  DataT read(const CoordT &Coords) const {
    return imageReadSamplerHostImpl<CoordT, DataT>(
        Coords, getSampler(), getSize(), getPitch(), getChannelType(),
        getChannelOrder(), getPtr(), getElementSize());
  }
#endif
};

} // namespace detail
} // namespace _V1
} // namespace sycl
