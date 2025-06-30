//==--------- node_impl.cpp - SYCL graph extension -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "node_impl.hpp"
#include "graph_impl.hpp"                              // for graph_impl
#include <sycl/ext/oneapi/experimental/graph/node.hpp> // for node

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

/// Takes a vector of weak_ptrs to node_impls and returns a vector of node
/// objects created from those impls, in the same order.
std::vector<node> createNodesFromImpls(
    const std::vector<std::weak_ptr<detail::node_impl>> &Impls) {
  std::vector<node> Nodes{};
  Nodes.reserve(Impls.size());

  for (std::weak_ptr<detail::node_impl> Impl : Impls) {
    Nodes.push_back(sycl::detail::createSyclObjFromImpl<node>(Impl.lock()));
  }

  return Nodes;
}

/// Takes a vector of shared_ptrs to node_impls and returns a vector of node
/// objects created from those impls, in the same order.
std::vector<node> createNodesFromImpls(
    const std::vector<std::shared_ptr<detail::node_impl>> &Impls) {
  std::vector<node> Nodes{};
  Nodes.reserve(Impls.size());

  for (std::shared_ptr<detail::node_impl> Impl : Impls) {
    Nodes.push_back(sycl::detail::createSyclObjFromImpl<node>(Impl));
  }

  return Nodes;
}
} // namespace detail

node_type node::get_type() const { return impl->MNodeType; }

std::vector<node> node::get_predecessors() const {
  return detail::createNodesFromImpls(impl->MPredecessors);
}

std::vector<node> node::get_successors() const {
  return detail::createNodesFromImpls(impl->MSuccessors);
}

node node::get_node_from_event(event nodeEvent) {
  auto EventImpl = sycl::detail::getSyclObjImpl(nodeEvent);
  auto GraphImpl = EventImpl->getCommandGraph();

  return sycl::detail::createSyclObjFromImpl<node>(
      GraphImpl->getNodeForEvent(EventImpl));
}

template <> __SYCL_EXPORT void node::update_nd_range<1>(nd_range<1> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_nd_range<2>(nd_range<2> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_nd_range<3>(nd_range<3> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_range<1>(range<1> Range) {
  impl->updateRange(Range);
}
template <> __SYCL_EXPORT void node::update_range<2>(range<2> Range) {
  impl->updateRange(Range);
}
template <> __SYCL_EXPORT void node::update_range<3>(range<3> Range) {
  impl->updateRange(Range);
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

size_t std::hash<sycl::ext::oneapi::experimental::node>::operator()(
    const sycl::ext::oneapi::experimental::node &Node) const {
  auto ID = sycl::detail::getSyclObjImpl(Node)->getID();
  return std::hash<decltype(ID)>()(ID);
}
