//==------------ image.hpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/image_impl.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>
#include <cstddef>
#include <sycl/ext/oneapi/accessor_property_list.hpp>

// sRGB Extension Support
#define SYCL_EXT_ONEAPI_SRGB 1

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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

/// Defines a shared image data.
///
/// Images can be 1-, 2-, and 3-dimensional. They have to be accessed using the
/// accessor class.
///
/// \sa sycl_api_acc
/// \sa sampler
///
/// \ingroup sycl_api
template <int Dimensions = 1, typename AllocatorT = cl::sycl::image_allocator>
class image {
public:
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        PropList);
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename detail::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        HostPointer, Order, Type, Range, Pitch,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            Allocator),
        PropList);
  }

  __SYCL2020_DEPRECATED("OpenCL interop APIs are deprecated")
  image(cl_mem ClMemObject, const context &SyclContext,
        event AvailableEvent = {}) {
    impl = std::make_shared<detail::image_impl<Dimensions>>(
        ClMemObject, SyclContext, AvailableEvent,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  /* -- common interface members -- */

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  bool operator==(const image &rhs) const { return impl == rhs.impl; }

  bool operator!=(const image &rhs) const { return !(*this == rhs); }

  /* -- property interface members -- */
  template <typename propertyT> bool has_property() const {
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
  }

  range<Dimensions> get_range() const { return impl->get_range(); }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  typename detail::enable_if_t<B, range<Dimensions - 1>> get_pitch() const {
    return impl->get_pitch();
  }

  // Returns the size of the image storage in bytes
  size_t get_size() const { return impl->getSize(); }

  // Returns the total number of elements in the image
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return impl->get_count(); }

  // Returns the allocator provided to the image
  AllocatorT get_allocator() const {
    return impl->template get_allocator<AllocatorT>();
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
    impl->set_final_data(finalData);
  }

  void set_write_back(bool flag = true) { impl->set_write_back(flag); }

private:
  std::shared_ptr<detail::image_impl<Dimensions>> impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <int Dimensions, typename AllocatorT>
struct hash<cl::sycl::image<Dimensions, AllocatorT>> {
  size_t operator()(const cl::sycl::image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<cl::sycl::detail::image_impl<Dimensions>>>()(
        cl::sycl::detail::getSyclObjImpl(I));
  }
};
} // namespace std
