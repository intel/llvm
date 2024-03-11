//==------------ item.hpp --- SYCL iteration item --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>            // for __SYCL_ASSUME_INT
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE, __SYC...
#include <sycl/detail/helpers.hpp>            // for Builder
#include <sycl/detail/item_base.hpp>          // for id, range, ItemBase
#include <sycl/exception.hpp> // for make_error_code, errc, exce...
#include <sycl/id.hpp>        // for id, item
#include <sycl/range.hpp>     // for range

#include <cstddef>     // for size_t
#include <type_traits> // for enable_if_t, conditional_t

namespace sycl {
inline namespace _V1 {

namespace detail {
class Builder;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH;
} // namespace detail

/// Identifies an instance of the function object executing at each point
/// in a range.
///
/// \ingroup sycl_api
template <int Dimensions = 1, bool with_offset = true> class item {
public:
  static constexpr int dimensions = Dimensions;

private:
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator __private_class(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class __private_class;

  template <bool B, typename T>
  using EnableIfT = std::conditional_t<B, T, __private_class>;
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
public:
  item() = delete;

  id<Dimensions> get_id() const { return MImpl.MIndex; }

  size_t __SYCL_ALWAYS_INLINE get_id(int Dimension) const {
    size_t Id = MImpl.MIndex[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE operator[](int Dimension) const {
    size_t Id = MImpl.MIndex[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  range<Dimensions> get_range() const { return MImpl.MExtent; }

  size_t __SYCL_ALWAYS_INLINE get_range(int Dimension) const {
    size_t Id = MImpl.MExtent[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  operator EnableIfT<Dimensions == 1, std::size_t>() const { return get_id(0); }
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  std::enable_if_t<has_offset, id<Dimensions>> get_offset() const {
    return MImpl.MOffset;
  }

  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  std::enable_if_t<has_offset, size_t> __SYCL_ALWAYS_INLINE
      get_offset(int Dimension) const {
    size_t Id = MImpl.MOffset[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  template <bool has_offset = with_offset>
  operator std::enable_if_t<!has_offset, item<Dimensions, true>>() const {
    return detail::Builder::createItem<Dimensions, true>(
        MImpl.MExtent, MImpl.MIndex, /*Offset*/ {});
  }

  size_t __SYCL_ALWAYS_INLINE get_linear_id() const {
    size_t Id = MImpl.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  item(const item &rhs) = default;

  item(item<Dimensions, with_offset> &&rhs) = default;

  item &operator=(const item &rhs) = default;

  item &operator=(item &&rhs) = default;

  bool operator==(const item &rhs) const { return rhs.MImpl == MImpl; }

  bool operator!=(const item &rhs) const { return rhs.MImpl != MImpl; }

protected:
  template <bool has_offset = with_offset>
  item(std::enable_if_t<has_offset, const range<Dimensions>> &extent,
       const id<Dimensions> &index, const id<Dimensions> &offset)
      : MImpl{extent, index, offset} {}

  template <bool has_offset = with_offset>
  item(std::enable_if_t<!has_offset, const range<Dimensions>> &extent,
       const id<Dimensions> &index)
      : MImpl{extent, index} {}

  friend class detail::Builder;

private:
  detail::ItemBase<Dimensions, with_offset> MImpl;
};

template <int Dims>
__SYCL_DEPRECATED("use sycl::ext::oneapi::experimental::this_item() instead")
item<Dims> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::Builder::getElement(detail::declptr<item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

namespace ext::oneapi::experimental {
template <int Dims> item<Dims> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(sycl::detail::declptr<item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
