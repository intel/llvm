//==--------- node.hpp --- SYCL graph extension ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"             // for graph_state
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/detail/impl_utils.hpp> // for getSyclObjImpl, createSyclObjFromImpl...
#include <sycl/detail/property_helper.hpp> // for PropertyWith...
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.hpp> // for graph properties classes
#include <sycl/nd_range.hpp> // for range, nd_range

#include <memory> // for shared_ptr
#include <vector> // for vector, hash

namespace sycl {
inline namespace _V1 {
class event;
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
// Forward declare ext::oneapi::experimental::detail classes
class node_impl;
} // namespace detail

enum class node_type {
  empty = 0,
  subgraph = 1,
  kernel = 2,
  memcpy = 3,
  memset = 4,
  memfill = 5,
  prefetch = 6,
  memadvise = 7,
  ext_oneapi_barrier = 8,
  host_task = 9,
  native_command = 10,
  async_malloc = 11,
  async_free = 12
};

/// Class representing a node in the graph, returned by command_graph::add().
class __SYCL_EXPORT node {
public:
  node() = delete;

  /// Get the type of command associated with this node.
  node_type get_type() const;

  /// Get a list of all the node dependencies of this node.
  std::vector<node> get_predecessors() const;

  /// Get a list of all nodes which depend on this node.
  std::vector<node> get_successors() const;

  /// Get the node associated with a SYCL event returned from a queue recording
  /// submission.
  static node get_node_from_event(event nodeEvent);

  /// Update the ND-Range of this node if it is a kernel execution node
  template <int Dimensions>
  void update_nd_range(nd_range<Dimensions> executionRange);

  /// Update the Range of this node if it is a kernel execution node
  template <int Dimensions> void update_range(range<Dimensions> executionRange);

  /// Common Reference Semantics
  friend bool operator==(const node &LHS, const node &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const node &LHS, const node &RHS) {
    return !operator==(LHS, RHS);
  }

private:
  node(const std::shared_ptr<detail::node_impl> &Impl) : impl(Impl) {}

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  std::shared_ptr<detail::node_impl> impl;
};

namespace property::node {
/// Property used to define dependent nodes when creating a new node with
/// command_graph::add().
class depends_on : public ::sycl::detail::PropertyWithData<
                       ::sycl::detail::GraphNodeDependencies> {
public:
  template <typename... NodeTN> depends_on(NodeTN... nodes) : MDeps{nodes...} {}

  const std::vector<::sycl::ext::oneapi::experimental::node> &
  get_dependencies() const {
    return MDeps;
  }

private:
  const std::vector<::sycl::ext::oneapi::experimental::node> MDeps;
};
} // namespace property::node

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
