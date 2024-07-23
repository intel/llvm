//==------------ image.hpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>                     // for placeholder
#include <sycl/aliases.hpp>                           // for cl_float, cl_half
#include <sycl/backend_types.hpp>                     // for backend, backe...
#include <sycl/buffer.hpp>                            // for range
#include <sycl/context.hpp>                           // for context
#include <sycl/detail/aligned_allocator.hpp>          // for aligned_allocator
#include <sycl/detail/backend_traits.hpp>             // for InteropFeature...
#include <sycl/detail/common.hpp>                     // for convertToArrayOfN
#include <sycl/detail/defines_elementary.hpp>         // for __SYCL2020_DEP...
#include <sycl/detail/export.hpp>                     // for __SYCL_EXPORT
#include <sycl/detail/impl_utils.hpp>                 // for getSyclObjImpl
#include <sycl/detail/owner_less_base.hpp>            // for OwnerLessBase
#include <sycl/detail/pi.h>                           // for pi_native_handle
#include <sycl/detail/stl_type_traits.hpp>            // for iterator_value...
#include <sycl/detail/sycl_mem_obj_allocator.hpp>     // for SYCLMemObjAllo...
#include <sycl/detail/type_list.hpp>                  // for is_contained
#include <sycl/event.hpp>                             // for event
#include <sycl/exception.hpp>                         // for make_error_code
#include <sycl/ext/oneapi/accessor_property_list.hpp> // for accessor_prope...
#include <sycl/property_list.hpp>                     // for property_list
#include <sycl/range.hpp>                             // for range, rangeTo...
#include <sycl/sampler.hpp>                           // for image_sampler
#include <sycl/types.hpp>                             // for vec

#include <cstddef>     // for size_t, nullptr_t
#include <functional>  // for function
#include <memory>      // for shared_ptr
#include <stdint.h>    // for uint8_t, uint32_t
#include <type_traits> // for enable_if_t
#include <variant>     // for hash

namespace sycl {
inline namespace _V1 {

// forward declarations
class handler;

template <int D, typename A> class image;

// 'friend'
template <backend Backend, int D, typename A>
std::enable_if_t<Backend == backend::ext_oneapi_level_zero, image<D, A>>
make_image(const backend_input_t<Backend, image<D, A>> &BackendObject,
           const context &TargetContext, event AvailableEvent = {});

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

// SYCL 2020 image_format
enum class image_format : unsigned int {
  r8g8b8a8_unorm = 0,
  r16g16b16a16_unorm = 1,
  r8g8b8a8_sint = 2,
  r16g16b16a16_sint = 3,
  r32b32g32a32_sint = 4,
  r8g8b8a8_uint = 5,
  r16g16b16a16_uint = 6,
  r32b32g32a32_uint = 7,
  r16b16g16a16_sfloat = 8,
  r32g32b32a32_sfloat = 9,
  b8g8r8a8_unorm = 10
};

using byte = unsigned char;

using image_allocator = detail::aligned_allocator<byte>;

namespace detail {

class image_impl;

// validImageDataT: cl_int4, cl_uint4, cl_float4, cl_half4
template <typename T>
using is_validImageDataT = typename detail::is_contained<
    T, type_list<vec<opencl::cl_int, 4>, vec<opencl::cl_uint, 4>,
                 vec<opencl::cl_float, 4>, vec<opencl::cl_half, 4>>>::type;

template <typename DataT>
using EnableIfImgAccDataT =
    typename std::enable_if_t<is_validImageDataT<DataT>::value, DataT>;

inline image_channel_type FormatChannelType(image_format Format) {
  switch (Format) {
  case image_format::r8g8b8a8_unorm:
  case image_format::b8g8r8a8_unorm:
    return image_channel_type::unorm_int8;
  case image_format::r16g16b16a16_unorm:
    return image_channel_type::unorm_int16;
  case image_format::r8g8b8a8_sint:
    return image_channel_type::signed_int8;
  case image_format::r16g16b16a16_sint:
    return image_channel_type::signed_int16;
  case image_format::r32b32g32a32_sint:
    return image_channel_type::signed_int32;
  case image_format::r8g8b8a8_uint:
    return image_channel_type::unsigned_int8;
  case image_format::r16g16b16a16_uint:
    return image_channel_type::unsigned_int16;
  case image_format::r32b32g32a32_uint:
    return image_channel_type::unsigned_int32;
  case image_format::r16b16g16a16_sfloat:
    return image_channel_type::fp16;
  case image_format::r32g32b32a32_sfloat:
    return image_channel_type::fp32;
  }
  throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                        "Unrecognized channel type.");
}

inline image_channel_order FormatChannelOrder(image_format Format) {
  switch (Format) {
  case image_format::r8g8b8a8_unorm:
  case image_format::r16g16b16a16_unorm:
  case image_format::r8g8b8a8_sint:
  case image_format::r16g16b16a16_sint:
  case image_format::r32b32g32a32_sint:
  case image_format::r8g8b8a8_uint:
  case image_format::r16g16b16a16_uint:
  case image_format::r32b32g32a32_uint:
  case image_format::r16b16g16a16_sfloat:
  case image_format::r32g32b32a32_sfloat:
    return image_channel_order::rgba;
  case image_format::b8g8r8a8_unorm:
    return image_channel_order::bgra;
  }
  throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                        "Unrecognized channel order.");
}

// The non-template base for the sycl::image class
class __SYCL_EXPORT image_plain {
protected:
  image_plain(const std::shared_ptr<detail::image_impl> &Impl) : impl{Impl} {}

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

  image_plain(const void *HostPointer, image_channel_order Order,
              image_channel_type Type, image_sampler Sampler,
              const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(const void *HostPointer, image_channel_order Order,
              image_channel_type Type, image_sampler Sampler,
              const range<3> &Range, const range<2> &Pitch,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(const std::shared_ptr<const void> &HostPointer,
              image_channel_order Order, image_channel_type Type,
              image_sampler Sampler, const range<3> &Range,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

  image_plain(const std::shared_ptr<const void> &HostPointer,
              image_channel_order Order, image_channel_type Type,
              image_sampler Sampler, const range<3> &Range,
              const range<2> &Pitch,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, const property_list &PropList);

#ifdef __SYCL_INTERNAL_API
  image_plain(cl_mem ClMemObject, const context &SyclContext,
              event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions);
#endif

  image_plain(pi_native_handle MemObject, const context &SyclContext,
              event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              uint8_t Dimensions, image_channel_order Order,
              image_channel_type Type, bool OwnNativeHandle,
              range<3> Range3WithOnes);

  template <typename propertyT> bool has_property() const noexcept {
    return getPropList().template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return getPropList().template get_property<propertyT>();
  }

  range<3> get_range() const;

  range<2> get_pitch() const;

  size_t get_size() const noexcept;

  size_t get_count() const noexcept;

  void set_final_data_internal();

  void set_final_data_internal(
      const std::function<void(const std::function<void(void *const Ptr)> &)>
          &FinalDataFunc);

  void set_write_back(bool flag);

  const std::unique_ptr<SYCLMemObjAllocator> &get_allocator_internal() const;

  size_t getElementSize() const;

  size_t getRowPitch() const;

  size_t getSlicePitch() const;

  image_sampler getSampler() const noexcept;

  image_channel_order getChannelOrder() const;

  image_channel_type getChannelType() const;

  void sampledImageConstructorNotification(const detail::code_location &CodeLoc,
                                           void *UserObj, const void *HostObj,
                                           uint32_t Dim, size_t Range[3],
                                           image_format Format,
                                           const image_sampler &Sampler);
  void sampledImageDestructorNotification(void *UserObj);

  void unsampledImageConstructorNotification(
      const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
      uint32_t Dim, size_t Range[3], image_format Format);
  void unsampledImageDestructorNotification(void *UserObj);

  std::shared_ptr<detail::image_impl> impl;

  const property_list &getPropList() const;
};

// Common base class for image implementations
template <int Dimensions, typename AllocatorT>
class image_common : public image_plain {
protected:
  // Use the same ctors as image_plain.
  using image_plain::image_plain;

public:
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
  template <bool IsMultiDim = (Dimensions > 1)>
  typename std::enable_if_t<IsMultiDim, range<Dimensions - 1>>
  get_pitch() const {
    return detail::convertToArrayOfN<Dimensions - 1, 0>(
        image_plain::get_pitch());
  }

  size_t size() const noexcept { return image_plain::get_count(); }

  // Returns the allocator provided to the image
  AllocatorT get_allocator() const {
    return image_plain::get_allocator_internal()
        ->template getAllocator<AllocatorT>();
  }
};

// Common base class for unsampled image implementations
template <int Dimensions, typename AllocatorT>
class unsampled_image_common : public image_common<Dimensions, AllocatorT> {
private:
  using common_base = typename detail::image_common<Dimensions, AllocatorT>;

protected:
  // Use the same ctors as image_plain.
  using common_base::image_common;

public:
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    this->set_final_data_internal(finalData);
  }

  void set_write_back(bool flag = true) { common_base::set_write_back(flag); }

private:
  void set_final_data_internal(std::nullptr_t) {
    common_base::set_final_data_internal();
  }

  template <template <typename WeakT> class WeakPtrT, typename WeakT>
  std::enable_if_t<
      std::is_convertible<WeakPtrT<WeakT>, std::weak_ptr<WeakT>>::value>
  set_final_data_internal(WeakPtrT<WeakT> FinalData) {
    std::weak_ptr<WeakT> TempFinalData(FinalData);
    this->set_final_data_internal(TempFinalData);
  }

  template <typename WeakT>
  void set_final_data_internal(std::weak_ptr<WeakT> FinalData) {
    common_base::set_final_data_internal(
        [FinalData](const std::function<void(void *const Ptr)> &F) {
          if (std::shared_ptr<WeakT> LockedFinalData = FinalData.lock())
            F(LockedFinalData.get());
        });
  }

  template <typename Destination>
  detail::EnableIfOutputPointerT<Destination>
  set_final_data_internal(Destination FinalData) {
    if (!FinalData)
      common_base::set_final_data_internal();
    else
      common_base::set_final_data_internal(
          [FinalData](const std::function<void(void *const Ptr)> &F) {
            F(FinalData);
          });
  }

  template <typename Destination>
  detail::EnableIfOutputIteratorT<Destination>
  set_final_data_internal(Destination FinalData) {
    const size_t Size = common_base::size();
    common_base::set_final_data_internal(
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
};

template <typename DataT, int Dims, access::mode AccMode,
          access::target AccTarget, access::placeholder IsPlaceholder>
class image_accessor;

} // namespace detail

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget>
class unsampled_image_accessor;

template <typename DataT, int Dimensions, access_mode AccessMode>
class host_unsampled_image_accessor;

template <typename DataT, int Dimensions, image_target AccessTarget>
class sampled_image_accessor;

template <typename DataT, int Dimensions> class host_sampled_image_accessor;

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
class image : public detail::unsampled_image_common<Dimensions, AllocatorT> {
private:
  using common_base =
      typename detail::unsampled_image_common<Dimensions, AllocatorT>;

public:
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {})
      : common_base(Order, Type, detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {})
      : common_base(
            Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : common_base(Order, Type, detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : common_base(
            Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, const property_list &PropList = {})
      : common_base(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range, AllocatorT Allocator,
        const property_list &PropList = {})
      : common_base(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {})
      : common_base(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  image(const void *HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {})
      : common_base(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : common_base(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(void *HostPointer, image_channel_order Order, image_channel_type Type,
        const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : common_base(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {}

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const property_list &PropList = {})
      : common_base(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {}

  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        AllocatorT Allocator, const property_list &PropList = {})
      : common_base(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        const property_list &PropList = {})
      : common_base(HostPointer, Order, Type,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {}

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  image(std::shared_ptr<void> &HostPointer, image_channel_order Order,
        image_channel_type Type, const range<Dimensions> &Range,
        const typename std::enable_if_t<B, range<Dimensions - 1>> &Pitch,
        AllocatorT Allocator, const property_list &PropList = {})
      : common_base(
            HostPointer, Order, Type, detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {}

#ifdef __SYCL_INTERNAL_API
  image(cl_mem ClMemObject, const context &SyclContext,
        event AvailableEvent = {})
      : common_base(ClMemObject, SyclContext, AvailableEvent,
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions) {}
#endif

  /* -- common interface members -- */

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  bool operator==(const image &rhs) const { return this->impl == rhs.impl; }

  bool operator!=(const image &rhs) const { return !(*this == rhs); }

  /* -- property interface members -- */
  template <typename propertyT> bool has_property() const noexcept {
    return common_base::template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return common_base::template get_property<propertyT>();
  }

  range<Dimensions> get_range() const {
    return detail::convertToArrayOfN<Dimensions, 0>(common_base::get_range());
  }

  /* Available only when: dimensions >1 */
  template <bool B = (Dimensions > 1)>
  typename std::enable_if_t<B, range<Dimensions - 1>> get_pitch() const {
    return detail::convertToArrayOfN<Dimensions - 1, 0>(
        common_base::get_pitch());
  }

  // Returns the size of the image storage in bytes
  size_t get_size() const { return common_base::get_size(); }

  // Returns the total number of elements in the image
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return common_base::size(); }

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

private:
  image(pi_native_handle MemObject, const context &SyclContext,
        event AvailableEvent, image_channel_order Order,
        image_channel_type Type, bool OwnNativeHandle, range<Dimensions> Range)
      : common_base(MemObject, SyclContext, AvailableEvent,
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, Order, Type, OwnNativeHandle,
                    detail::convertToArrayOfN<3, 1>(Range)) {}

  // This utility api is currently used by accessor to get the element size of
  // the image. Element size is dependent on num of channels and channel type.
  // This information is not accessible from the image using any public API.
  size_t getElementSize() const { return common_base::getElementSize(); }

  size_t getRowPitch() const { return common_base::getRowPitch(); }

  size_t getSlicePitch() const { return common_base::getSlicePitch(); }

  image_channel_order getChannelOrder() const {
    return common_base::getChannelOrder();
  }

  image_channel_type getChannelType() const {
    return common_base::getChannelType();
  }

  // Declare make_image as a friend function
  template <backend Backend, int D, typename A>
  friend std::enable_if_t<
      detail::InteropFeatureSupportMap<Backend>::MakeImage == true &&
          Backend != backend::ext_oneapi_level_zero,
      image<D, A>>
  make_image(
      const typename backend_traits<Backend>::template input_type<image<D, A>>
          &BackendObject,
      const context &TargetContext, event AvailableEvent);

  template <backend Backend, int D, typename A>
  friend std::enable_if_t<Backend == backend::ext_oneapi_level_zero,
                          image<D, A>>
  make_image(const backend_input_t<Backend, image<D, A>> &BackendObject,
             const context &TargetContext, event AvailableEvent);

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPlaceholder,
            typename PropertyListT>
  friend class accessor;

  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder IsPlaceholder>
  friend class detail::image_accessor;
};

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class unsampled_image
    : public detail::unsampled_image_common<Dimensions, AllocatorT>,
      public detail::OwnerLessBase<unsampled_image<Dimensions, AllocatorT>> {
private:
  using common_base =
      typename detail::unsampled_image_common<Dimensions, AllocatorT>;

  unsampled_image(const std::shared_ptr<detail::image_impl> &Impl)
      : common_base{Impl} {}

public:
  unsampled_image(
      image_format Format, const range<Dimensions> &Range,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), nullptr, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  unsampled_image(
      image_format Format, const range<Dimensions> &Range, AllocatorT Allocator,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), nullptr, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      image_format Format, const range<Dimensions> &Range,
      const range<Dimensions - 1> &Pitch, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), nullptr, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      image_format Format, const range<Dimensions> &Range,
      const range<Dimensions - 1> &Pitch, AllocatorT Allocator,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), nullptr, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  unsampled_image(
      void *HostPointer, image_format Format, const range<Dimensions> &Range,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  unsampled_image(
      void *HostPointer, image_format Format, const range<Dimensions> &Range,
      AllocatorT Allocator, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            HostPointer, detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      void *HostPointer, image_format Format, const range<Dimensions> &Range,
      const range<Dimensions - 1> &Pitch, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      void *HostPointer, image_format Format, const range<Dimensions> &Range,
      const range<Dimensions - 1> &Pitch, AllocatorT Allocator,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            HostPointer, detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer, Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  unsampled_image(
      std::shared_ptr<void> &HostPointer, image_format Format,
      const range<Dimensions> &Range, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  unsampled_image(
      std::shared_ptr<void> &HostPointer, image_format Format,
      const range<Dimensions> &Range, AllocatorT Allocator,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            HostPointer, detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      std::shared_ptr<void> &HostPointer, image_format Format,
      const range<Dimensions> &Range, const range<Dimensions - 1> &Pitch,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format),
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList, /*IsConstPtr*/ false) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  unsampled_image(
      std::shared_ptr<void> &HostPointer, image_format Format,
      const range<Dimensions> &Range, const range<Dimensions - 1> &Pitch,
      AllocatorT Allocator, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(
            HostPointer, detail::FormatChannelOrder(Format),
            detail::FormatChannelType(Format),
            detail::convertToArrayOfN<3, 1>(Range),
            detail::convertToArrayOfN<2, 0>(Pitch),
            std::make_unique<
                detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(Allocator),
            Dimensions, PropList, /*IsConstPtr*/ false) {
    common_base::unsampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format);
  }

  /* -- common interface members -- */

  unsampled_image(const unsampled_image &rhs) = default;

  unsampled_image(unsampled_image &&rhs) = default;

  unsampled_image &operator=(const unsampled_image &rhs) = default;

  unsampled_image &operator=(unsampled_image &&rhs) = default;

  ~unsampled_image() {
    try {
      common_base::unsampledImageDestructorNotification(
          (void *)this->impl.get());
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~unsampled_image", e);
    }
  }

  bool operator==(const unsampled_image &rhs) const {
    return this->impl == rhs.impl;
  }

  bool operator!=(const unsampled_image &rhs) const { return !(*this == rhs); }

  size_t byte_size() const noexcept { return common_base::get_size(); }

  using common_base::size;

  template <typename DataT,
            access_mode AccessMode = (std::is_const_v<DataT>
                                          ? access_mode::read
                                          : access_mode::read_write),
            image_target AccessTarget = image_target::device>
  unsampled_image_accessor<DataT, Dimensions, AccessMode, AccessTarget>
  get_access(
      handler &CommandGroupHandlerRef, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return {*this, CommandGroupHandlerRef, PropList, CodeLoc};
  }

  template <typename DataT,
            access_mode AccessMode = (std::is_const_v<DataT>
                                          ? access_mode::read
                                          : access_mode::read_write)>
  host_unsampled_image_accessor<DataT, Dimensions, AccessMode> get_host_access(
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return {*this, PropList, CodeLoc};
  }

private:
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <typename DataT, int Dims, access_mode AccessMode>
  friend class host_unsampled_image_accessor;

  template <typename DataT, int Dims, access_mode AccessMode,
            image_target AccessTarget>
  friend class unsampled_image_accessor;
};

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class sampled_image
    : public detail::image_common<Dimensions, AllocatorT>,
      public detail::OwnerLessBase<sampled_image<Dimensions, AllocatorT>> {
private:
  using common_base = typename detail::image_common<Dimensions, AllocatorT>;

  sampled_image(const std::shared_ptr<detail::image_impl> &Impl)
      : common_base{Impl} {}

public:
  sampled_image(
      const void *HostPointer, image_format Format, image_sampler Sampler,
      const range<Dimensions> &Range, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format), Sampler,
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::sampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), nullptr, Dimensions,
        detail::rangeToArray(Range).data(), Format, Sampler);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  sampled_image(
      const void *HostPointer, image_format Format, image_sampler Sampler,
      const range<Dimensions> &Range, const range<Dimensions - 1> &Pitch,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format), Sampler,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::sampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer, Dimensions,
        detail::rangeToArray(Range).data(), Format, Sampler);
  }

  sampled_image(
      std::shared_ptr<const void> &HostPointer, image_format Format,
      image_sampler Sampler, const range<Dimensions> &Range,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format), Sampler,
                    detail::convertToArrayOfN<3, 1>(Range),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::sampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format, Sampler);
  }

  template <bool IsMultiDim = (Dimensions > 1),
            typename = std::enable_if_t<IsMultiDim>>
  sampled_image(
      std::shared_ptr<const void> &HostPointer, image_format Format,
      image_sampler Sampler, const range<Dimensions> &Range,
      const range<Dimensions - 1> &Pitch, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : common_base(HostPointer, detail::FormatChannelOrder(Format),
                    detail::FormatChannelType(Format), Sampler,
                    detail::convertToArrayOfN<3, 1>(Range),
                    detail::convertToArrayOfN<2, 0>(Pitch),
                    std::make_unique<
                        detail::SYCLMemObjAllocatorHolder<AllocatorT, byte>>(),
                    Dimensions, PropList) {
    common_base::sampledImageConstructorNotification(
        CodeLoc, (void *)this->impl.get(), HostPointer.get(), Dimensions,
        detail::rangeToArray(Range).data(), Format, Sampler);
  }

  /* -- common interface members -- */

  sampled_image(const sampled_image &rhs) = default;

  sampled_image(sampled_image &&rhs) = default;

  sampled_image &operator=(const sampled_image &rhs) = default;

  sampled_image &operator=(sampled_image &&rhs) = default;

  ~sampled_image() {
    try {
      common_base::sampledImageDestructorNotification((void *)this->impl.get());
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~sampled_image", e);
    }
  }

  bool operator==(const sampled_image &rhs) const {
    return this->impl == rhs.impl;
  }

  bool operator!=(const sampled_image &rhs) const { return !(*this == rhs); }

  size_t byte_size() const noexcept { return common_base::get_size(); }

  template <typename DataT, image_target AccessTarget = image_target::device>
  sampled_image_accessor<DataT, Dimensions, AccessTarget> get_access(
      handler &CommandGroupHandlerRef, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return {*this, CommandGroupHandlerRef, PropList, CodeLoc};
  }

  template <typename DataT>
  host_sampled_image_accessor<DataT, Dimensions> get_host_access(
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return {*this, PropList, CodeLoc};
  }

private:
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <typename DataT, int Dims> friend class host_sampled_image_accessor;

  template <typename DataT, int Dims, image_target AccessTarget>
  friend class sampled_image_accessor;
};

} // namespace _V1
} // namespace sycl

namespace std {
template <int Dimensions, typename AllocatorT>
struct hash<sycl::image<Dimensions, AllocatorT>> {
  size_t operator()(const sycl::image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<sycl::detail::image_impl>>()(
        sycl::detail::getSyclObjImpl(I));
  }
};

template <int Dimensions, typename AllocatorT>
struct hash<sycl::unsampled_image<Dimensions, AllocatorT>> {
  size_t
  operator()(const sycl::unsampled_image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<sycl::detail::image_impl>>()(
        sycl::detail::getSyclObjImpl(I));
  }
};

template <int Dimensions, typename AllocatorT>
struct hash<sycl::sampled_image<Dimensions, AllocatorT>> {
  size_t
  operator()(const sycl::sampled_image<Dimensions, AllocatorT> &I) const {
    return hash<std::shared_ptr<sycl::detail::image_impl>>()(
        sycl::detail::getSyclObjImpl(I));
  }
};
} // namespace std
