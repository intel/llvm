//==------------ item.hpp --- SYCL iteration item --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/item_base.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>

#include <cstddef>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class Builder;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH;
}
template <int dimensions> class id;
template <int dimensions> class range;

/// Identifies an instance of the function object executing at each point
/// in a range.
///
/// \ingroup sycl_api
template <int dimensions = 1, bool with_offset = true> class item {
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator __private_class(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class __private_class;

  template <bool B, typename T>
  using EnableIfT = detail::conditional_t<B, T, __private_class>;
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
public:
  item() = delete;

  id<dimensions> get_id() const { return MImpl.MIndex; }

  size_t __SYCL_ALWAYS_INLINE get_id(int dimension) const {
    size_t Id = MImpl.MIndex[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE operator[](int dimension) const {
    size_t Id = MImpl.MIndex[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  range<dimensions> get_range() const { return MImpl.MExtent; }

  size_t __SYCL_ALWAYS_INLINE get_range(int dimension) const {
    size_t Id = MImpl.MExtent[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  operator EnableIfT<dimensions == 1, std::size_t>() const { return get_id(0); }
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  detail::enable_if_t<has_offset, id<dimensions>> get_offset() const {
    return MImpl.MOffset;
  }

  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  detail::enable_if_t<has_offset, size_t> __SYCL_ALWAYS_INLINE
      get_offset(int dimension) const {
    size_t Id = MImpl.MOffset[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  template <bool has_offset = with_offset>
  operator detail::enable_if_t<!has_offset, item<dimensions, true>>() const {
    return detail::Builder::createItem<dimensions, true>(
        MImpl.MExtent, MImpl.MIndex, /*Offset*/ {});
  }

  size_t __SYCL_ALWAYS_INLINE get_linear_id() const {
    size_t Id = MImpl.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  item(const item &rhs) = default;

  item(item<dimensions, with_offset> &&rhs) = default;

  item &operator=(const item &rhs) = default;

  item &operator=(item &&rhs) = default;

  bool operator==(const item &rhs) const { return rhs.MImpl == MImpl; }

  bool operator!=(const item &rhs) const { return rhs.MImpl != MImpl; }

protected:
  template <bool has_offset = with_offset>
  item(detail::enable_if_t<has_offset, const range<dimensions>> &extent,
       const id<dimensions> &index, const id<dimensions> &offset)
      : MImpl{extent, index, offset} {}

  template <bool has_offset = with_offset>
  item(detail::enable_if_t<!has_offset, const range<dimensions>> &extent,
       const id<dimensions> &index)
      : MImpl{extent, index} {}

  friend class detail::Builder;

private:
  // Friend to get access to private method set_allowed_range().
  template <typename, int, typename> friend class detail::RoundedRangeKernel;
  template <typename, int, typename>
  friend class detail::RoundedRangeKernelWithKH;
  void set_allowed_range(const range<dimensions> rnwi) { MImpl.MExtent = rnwi; }

  detail::ItemBase<dimensions, with_offset> MImpl;
};

template <int Dims>
__SYCL_DEPRECATED("use sycl::ext::oneapi::experimental::this_item() instead")
item<Dims> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::Builder::getElement(detail::declptr<item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host device");
#endif
}

namespace ext {
namespace oneapi {
namespace experimental {
template <int Dims> item<Dims> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(sycl::detail::declptr<item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host device");
#endif
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
