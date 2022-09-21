//==------------ image.hpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/aligned_allocator.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/sycl_mem_obj_allocator.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/stl.hpp>
#include <sycl/types.hpp>

#include <cstddef>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;

enum class image_channel_order : unsigned int {
  a = 0,
  r = 1,
  rx = 2,
  rg = 3,
  rgx = 4,
  ra = 5,
  rgb = 6,
  rgbx = 7,
  rgba = 8,
  argb = 9,
  bgra = 10,
  intensity = 11,
  luminance = 12,
  abgr = 13,
  ext_oneapi_srgba = 14 // OpenCL 2.0
};

enum class image_channel_type : unsigned int {
  snorm_int8 = 0,
  snorm_int16 = 1,
  unorm_int8 = 2,
  unorm_int16 = 3,
  unorm_short_565 = 4,
  unorm_short_555 = 5,
  unorm_int_101010 = 6,
  signed_int8 = 7,
  signed_int16 = 8,
  signed_int32 = 9,
  unsigned_int8 = 10,
  unsigned_int16 = 11,
  unsigned_int32 = 12,
  fp16 = 13,
  fp32 = 14
};

using byte = unsigned char;

using image_allocator = detail::aligned_allocator<byte>;

namespace detail {

class image_impl;

// validImageDataT: cl_int4, cl_uint4, cl_float4, cl_half4
template <typename T>
using is_validImageDataT = typename detail::is_contained<
    T, type_list<cl_int4, cl_uint4, cl_float4, cl_half4>>::type;

template <typename DataT>
using EnableIfImgAccDataT =
    typename detail::enable_if_t<is_validImageDataT<DataT>::value, DataT>;

// The non-template base for the sycl::image class
class __SYCL_EXPORT image_plain {
protected:
  image_plain(image_channel_order Order, image_channel_type Type,
              const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(image_channel_order Order, image_channel_type Type,
              const range<3> &Range, const range<2> &Pitch,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(void *HostPointer, image_channel_order Order,
              image_channel_type Type, const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(const void *HostPointer, image_channel_order Order,
              image_channel_type Type, const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(void *HostPointer, image_channel_order Order,
              image_channel_type Type, const range<3> &Range,
              const range<2> &Pitch,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(const std::shared_ptr<const void> &HostPointer,
              image_channel_order Order, image_channel_type Type,
              const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList,
              bool IsConstPtr);

  image_plain(const std::shared_ptr<const void> &HostPointer,
              image_channel_order Order, image_channel_type Type,
              const range<3> &Range, const range<2> &Pitch,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList,
              bool IsConstPtr);

#ifdef __SYCL_INTERNAL_API
  image_plain(cl_mem ClMemObject, const context &SyclContext,
              event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions);
#endif

  template <typename propertyT> bool has_property() const noexcept;

  template <typename propertyT> propertyT get_property() const;

  range<3> get_range() const;

  range<2> get_pitch() const;

  size_t get_size() const;

  size_t get_count() const;

  void set_final_data_internal();

  void set_final_data_internal(
      const std::function<void(const std::function<void(void *const Ptr)> &)>
          &FinalDataFunc);

  void set_write_back(bool flag);

  const std::unique_ptr<SYCLMemObjAllocator> &get_allocator_internal() const;

  size_t getElementSize() const;

  size_t getRowPitch() const;

  size_t getSlicePitch() const;

  image_channel_order getChannelOrder() const;

  image_channel_type getChannelType() const;

  std::shared_ptr<detail::image_impl> impl;
};

template <typename DataT, int Dims, access::mode AccMode,
          access::target AccTarget, access::placeholder IsPlaceholder>
class image_accessor;

} // namespace detail

/// Defines a shared image data.
///
/// Images can be 1-, 2-, and 3-dimensional. They have to be accessed using the
/// accessor class.
///
/// \sa sycl_api_acc
/// \sa sampler
///
/// \ingroup sycl_api
template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class image : public detail::image_plain {

public:
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {})
      : image_plain(Order, Type, detail::convertToArrayOfN<3, 1>(Range),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {})
      : image_plain(
            Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : image_plain(Order, Type, detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : image_plain(
            Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {})
      : image_plain(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {})
      : image_plain(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {})
      : image_plain(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {})
      : image_plain(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : image_plain(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : image_plain(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {})
      : image_plain(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {}

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {})
      : image_plain(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : image_plain(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : image_plain(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            make_unique_ptr<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {}

#ifdef __SYCL_INTERNAL_API
  image(cl_mem ClMemObject, const context &SyclContext,
        event AvailableEvent = {})
      : image_plain(ClMemObject, SyclContext, AvailableEvent,
                    make_unique_ptr<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions) {}
#endif

  /* -- common interface members -- */

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  bool operator==(const image &rhs) const { return impl == rhs.impl; }

  bool operator!=(const image &rhs) const { return !(*this == rhs); }

  /* -- property interface members -- */
  template <typename propertyT> bool has_property() const noexcept {
    return image_plain::template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return image_plain::get_property<propertyT>();
  }

  range<Dimensions> get_range() const {
    return detail::convertToArrayOfN<Dimensions, 0>(image_plain::get_range());
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  typename detail::enable_if_t<B, range<Dimensions - 1>> get_pitch() const {
    return detail::convertToArrayOfN<Dimensions - 1, 0>(
        image_plain::get_pitch());
  }

  // Returns the size of the image storage in bytes
  size_t get_size() const { return image_plain::get_size(); }

  // Returns the total number of elements in the image
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return image_plain::get_count(); }

  // Returns the allocator provided to the image
  AllocatorT get_allocator() const {
    return image_plain::get_allocator_internal()
        ->template getAllocator<AllocatorT>();
  }

  template <typename DataT, access::mode AccessMode>
  accessor<detail::EnableIfImgAccDataT<DataT>, Dimensions, AccessMode,
           access::target::image, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access(handler &commandGroupHandler) {
    return accessor<DataT, Dimensions, AccessMode, access::target::image,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this,
                                                           commandGroupHandler);
  }

  template <typename DataT, access::mode AccessMode>
  accessor<detail::EnableIfImgAccDataT<DataT>, Dimensions, AccessMode,
           access::target::host_image, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access() {
    return accessor<DataT, Dimensions, AccessMode, access::target::host_image,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this);
  }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    this->set_final_data_internal(finalData);
  }

  void set_final_data_internal(std::nullptr_t) {
    image_plain::set_final_data_internal();
  }

  template <template <typename WeakT> class WeakPtrT, typename WeakT>
  detail::enable_if_t<
      std::is_convertible<WeakPtrT<WeakT>, std::weak_ptr<WeakT>>::value>
  set_final_data_internal(WeakPtrT<WeakT> FinalData) {
    std::weak_ptr<WeakT> TempFinalData(FinalData);
    this->set_final_data_internal(TempFinalData);
  }

  template <typename WeakT>
  void set_final_data_internal(std::weak_ptr<WeakT> FinalData) {
    image_plain::set_final_data_internal(
        [FinalData](const std::function<void(void *const Ptr)> &F) {
          if (std::shared_ptr<WeakT> LockedFinalData = FinalData.lock())
            F(LockedFinalData.get());
        });
  }

  template <typename Destination>
  detail::EnableIfOutputPointerT<Destination>
  set_final_data_internal(Destination FinalData) {
    if (!FinalData)
      image_plain::set_final_data_internal();
    else
      image_plain::set_final_data_internal(
          [FinalData](const std::function<void(void *const Ptr)> &F) {
            F(FinalData);
          });
  }

  template <typename Destination>
  detail::EnableIfOutputIteratorT<Destination>
  set_final_data_internal(Destination FinalData) {
    const size_t Size = size();
    image_plain::set_final_data_internal(
        [FinalData, Size](const std::function<void(void *const Ptr)> &F) {
          using DestinationValueT = detail::iterator_value_type_t<Destination>;
          // TODO if Destination is ContiguousIterator then don't create
          // ContiguousStorage. updateHostMemory works only with pointer to
          // continuous data.
          std::unique_ptr<DestinationValueT[]> ContiguousStorage(
              new DestinationValueT[Size]);
          F(ContiguousStorage.get());
          std::copy(ContiguousStorage.get(), ContiguousStorage.get() + Size,
                    FinalData);
        });
  }

  void set_write_back(bool flag = true) { image_plain::set_write_back(flag); }

private:
  // This utility api is currently used by accessor to get the element size of
  // the image. Element size is dependent on num of channels and channel type.
  // This information is not accessible from the image using any public API.
  size_t getElementSize() const { return image_plain::getElementSize(); }

  size_t getRowPitch() const { return image_plain::getRowPitch(); }

  size_t getSlicePitch() const { return image_plain::getSlicePitch(); }

  image_channel_order getChannelOrder() const {
    return image_plain::getChannelOrder();
  }

  image_channel_type getChannelType() const {
    return image_plain::getChannelType();
  }

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPlaceholder,
            typename PropertyListT>
  friend class accessor;

  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPlaceholder>
  friend class detail::image_accessor;
};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace std {
template <int Dimensions, typename AllocatorT>
struct hash<sycl::image<Dimensions, AllocatorT>> {
  size_t operator()(const sycl::image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<sycl::detail::image_impl>>()(
        sycl::detail::getSyclObjImpl(I));
  }
};
} // namespace std
