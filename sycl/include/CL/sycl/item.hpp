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
}
template <int dimensions> class id;
template <int dimensions> class range;

/// Identifies an instance of the function object executing at each point
/// in a range.
///
/// \ingroup sycl_api
template <int dimensions = 1, bool with_offset = true> class item {
public:
  item() = delete;

  id<dimensions> get_id() const { return MImpl.MIndex; }

  size_t ALWAYS_INLINE get_id(int dimension) const {
    size_t Id = MImpl.MIndex[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t ALWAYS_INLINE operator[](int dimension) const {
    size_t Id = MImpl.MIndex[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  range<dimensions> get_range() const { return MImpl.MExtent; }

  size_t ALWAYS_INLINE get_range(int dimension) const {
    size_t Id = MImpl.MExtent[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  template <bool has_offset = with_offset>
  detail::enable_if_t<has_offset, id<dimensions>> get_offset() const {
    return MImpl.MOffset;
  }

  template <bool has_offset = with_offset>
  detail::enable_if_t<has_offset, size_t>
      ALWAYS_INLINE get_offset(int dimension) const {
    size_t Id = MImpl.MOffset[dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  template <bool has_offset = with_offset>
  operator detail::enable_if_t<!has_offset, item<dimensions, true>>() const {
    return detail::Builder::createItem<dimensions, true>(
        MImpl.MExtent, MImpl.MIndex, /*Offset*/ {});
  }

  size_t ALWAYS_INLINE get_linear_id() const {
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
  detail::ItemBase<dimensions, with_offset> MImpl;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
