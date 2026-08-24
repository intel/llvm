//==-------------- h_item.hpp - SYCL standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/helpers.hpp>   // for Builder, group
#include <sycl/detail/item_base.hpp> // for id, range
#include <sycl/id.hpp>               // for id
#include <sycl/item.hpp>             // for item
#include <sycl/range.hpp>            // for range

#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {

namespace detail {
class Builder;
}

/// Identifies an instance of a group::parallel_for_work_item function object
/// executing at each point in a local range passed to a parallel_for_work_item
/// call or to the corresponding parallel_for_work_group call.
///
/// \ingroup sycl_api
template <int Dimensions> class h_item {
public:
  static constexpr int dimensions = Dimensions;

  h_item() = delete;

  h_item(const h_item &hi) = default;

  h_item &operator=(const h_item &hi) = default;

  /* -- public interface members -- */
  item<Dimensions, false> get_global() const { return globalItem; }

  item<Dimensions, false> get_local() const { return get_logical_local(); }

  item<Dimensions, false> get_logical_local() const { return logicalLocalItem; }

  item<Dimensions, false> get_physical_local() const { return localItem; }

  range<Dimensions> get_global_range() const {
    return get_global().get_range();
  }

  size_t get_global_range(int Dimension) const {
    return get_global().get_range(Dimension);
  }

  id<Dimensions> get_global_id() const { return get_global().get_id(); }

  size_t get_global_id(int Dimension) const {
    return get_global().get_id(Dimension);
  }

  range<Dimensions> get_local_range() const { return get_local().get_range(); }

  size_t get_local_range(int Dimension) const {
    return get_local().get_range(Dimension);
  }

  id<Dimensions> get_local_id() const { return get_local().get_id(); }

  size_t get_local_id(int Dimension) const {
    return get_local().get_id(Dimension);
  }

  range<Dimensions> get_logical_local_range() const {
    return get_logical_local().get_range();
  }

  size_t get_logical_local_range(int Dimension) const {
    return get_logical_local().get_range(Dimension);
  }

  id<Dimensions> get_logical_local_id() const {
    return get_logical_local().get_id();
  }

  size_t get_logical_local_id(int Dimension) const {
    return get_logical_local().get_id(Dimension);
  }

  range<Dimensions> get_physical_local_range() const {
    return get_physical_local().get_range();
  }

  size_t get_physical_local_range(int Dimension) const {
    return get_physical_local().get_range(Dimension);
  }

  id<Dimensions> get_physical_local_id() const {
    return get_physical_local().get_id();
  }

  size_t get_physical_local_id(int Dimension) const {
    return get_physical_local().get_id(Dimension);
  }

  bool operator==(const h_item &rhs) const {
    return (rhs.localItem == localItem) && (rhs.globalItem == globalItem) &&
           (rhs.logicalLocalItem == logicalLocalItem);
  }

  bool operator!=(const h_item &rhs) const { return !((*this) == rhs); }

protected:
  friend class detail::Builder;
  friend class group<Dimensions>;
  h_item(const item<Dimensions, false> &GL, const item<Dimensions, false> &L,
         const range<Dimensions> &flexLocalRange)
      : globalItem(GL), localItem(L),
        logicalLocalItem(detail::Builder::createItem<Dimensions, false>(
            flexLocalRange, L.get_id())) {}

  h_item(const item<Dimensions, false> &GL, const item<Dimensions, false> &L)
      : globalItem(GL), localItem(L),
        logicalLocalItem(detail::Builder::createItem<Dimensions, false>(
            localItem.get_range(), localItem.get_id())) {}

  void setLogicalLocalID(const id<Dimensions> &ID) {
    detail::Builder::updateItemIndex(logicalLocalItem, ID);
  }

private:
  /// Describles physical workgroup range and current \c h_item position in it.
  item<Dimensions, false> localItem;
  /// Describles global range and current \c h_item position in it.
  item<Dimensions, false> globalItem;
  /// Describles logical flexible range and current \c h_item position in it.
  item<Dimensions, false> logicalLocalItem;
};

} // namespace _V1
} // namespace sycl
