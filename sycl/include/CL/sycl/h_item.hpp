//==-------------- h_item.hpp - SYCL standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/item.hpp>
#include <CL/sycl/range.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {
class Builder;
}

template <int dimensions> class h_item {
public:
  h_item() = delete;

  h_item(const h_item &hi) = default;

  h_item &operator=(const h_item &hi) = default;

  /* -- public interface members -- */
  item<dimensions, false> get_global() const { return globalItem; }

  item<dimensions, false> get_local() const { return get_logical_local(); }

  item<dimensions, false> get_logical_local() const { return logicalLocalItem; }

  item<dimensions, false> get_physical_local() const { return localItem; }

  range<dimensions> get_global_range() const {
    return get_global().get_range();
  }

  size_t get_global_range(int dimension) const {
    return get_global().get_range(dimension);
  }

  id<dimensions> get_global_id() const { return get_global().get_id(); }

  size_t get_global_id(int dimension) const {
    return get_global().get_id(dimension);
  }

  range<dimensions> get_local_range() const { return get_local().get_range(); }

  size_t get_local_range(int dimension) const {
    return get_local().get_range(dimension);
  }

  id<dimensions> get_local_id() const { return get_local().get_id(); }

  size_t get_local_id(int dimension) const {
    return get_local().get_id(dimension);
  }

  range<dimensions> get_logical_local_range() const {
    return get_logical_local().get_range();
  }

  size_t get_logical_local_range(int dimension) const {
    return get_logical_local().get_range(dimension);
  }

  id<dimensions> get_logical_local_id() const {
    return get_logical_local().get_id();
  }

  size_t get_logical_local_id(int dimension) const {
    return get_logical_local().get_id(dimension);
  }

  range<dimensions> get_physical_local_range() const {
    return get_physical_local().get_range();
  }

  size_t get_physical_local_range(int dimension) const {
    return get_physical_local().get_range(dimension);
  }

  id<dimensions> get_physical_local_id() const {
    return get_physical_local().get_id();
  }

  size_t get_physical_local_id(int dimension) const {
    return get_physical_local().get_id(dimension);
  }

  bool operator==(const h_item &rhs) const {
    return (rhs.localItem == localItem) && (rhs.globalItem == globalItem) &&
           (rhs.logicalLocalItem == logicalLocalItem);
  }

  bool operator!=(const h_item &rhs) const { return !((*this) == rhs); }

protected:
  friend class detail::Builder;
  friend class group<dimensions>;
  h_item(const item<dimensions, false> &GL, const item<dimensions, false> &L,
         const range<dimensions> &flexLocalRange)
      : globalItem(GL), localItem(L),
        logicalLocalItem(detail::Builder::createItem<dimensions, false>(
            flexLocalRange, L.get_id())) {}

  h_item(const item<dimensions, false> &GL, const item<dimensions, false> &L)
      : globalItem(GL), localItem(L),
        logicalLocalItem(detail::Builder::createItem<dimensions, false>(
            localItem.get_range(), localItem.get_id())) {}

  void setLogicalLocalID(const id<dimensions> &ID) {
    detail::Builder::updateItemIndex(logicalLocalItem, ID);
  }

private:
  /// Describles physical workgroup range and current \c h_item position in it.
  item<dimensions, false> localItem;
  /// Describles global range and current \c h_item position in it.
  item<dimensions, false> globalItem;
  /// Describles logical flexible range and current \c h_item position in it.
  item<dimensions, false> logicalLocalItem;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
