//==--------- graph_impl.cpp - SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/graph_impl.hpp>
#include <detail/queue_impl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
struct queue_impl;
using queue_ptr = std::shared_ptr<queue_impl>;
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

void graph_impl::exec(sycl::detail::queue_ptr q) {
  if (MSchedule.empty()) {
    for (auto n : MRoots) {
      n->topology_sort(MSchedule);
    }
  }
  for (auto n : MSchedule)
    n->exec(q);
}

void graph_impl::exec_and_wait(sycl::detail::queue_ptr q) {
  bool isSubGraph = q->getIsGraphSubmitting();
  if (!isSubGraph) {
    q->setIsGraphSubmitting(true);
  }
  if (MFirst) {
    exec(q);
    MFirst = false;
  }
  if (!isSubGraph) {
    q->setIsGraphSubmitting(false);
    q->wait();
  }
}

void graph_impl::add_root(node_ptr n) {
  MRoots.insert(n);
  for (auto n : MSchedule)
    n->MScheduled = false;
  MSchedule.clear();
}

void graph_impl::remove_root(node_ptr n) {
  MRoots.erase(n);
  for (auto n : MSchedule)
    n->MScheduled = false;
  MSchedule.clear();
}

template <typename T>
node_ptr graph_impl::add(graph_ptr impl, T cgf,
                         const std::vector<node_ptr> &dep) {
  node_ptr nodeImpl = std::make_shared<node_impl>(impl, cgf);
  if (!dep.empty()) {
    for (auto n : dep) {
      n->register_successor(nodeImpl); // register successor
      this->remove_root(nodeImpl);     // remove receiver from root node
                                       // list
    }
  } else {
    this->add_root(nodeImpl);
  }
  return nodeImpl;
}

void node_impl::exec(sycl::detail::queue_ptr q) {
  std::vector<sycl::event> deps;
  for (auto i : MPredecessors)
    deps.push_back(i->get_event());

  const sycl::detail::code_location CodeLoc;
  MEvent = q->submit(wrapper{MBody, deps}, q, CodeLoc);
}
} // namespace detail

template <>
command_graph<graph_state::modifiable>::command_graph(
    const sycl::property_list &)
    : impl(std::make_shared<detail::graph_impl>()) {}

template <>
node command_graph<graph_state::modifiable>::add_impl(
    std::function<void(handler &)> cgf, const std::vector<node> &dep) {
  std::vector<detail::node_ptr> depImpls;
  for (auto &d : dep) {
    depImpls.push_back(sycl::detail::getSyclObjImpl(d));
  }

  auto nodeImpl = impl->add(impl, cgf, depImpls);
  return sycl::detail::createSyclObjFromImpl<node>(nodeImpl);
}

template <>
void command_graph<graph_state::modifiable>::make_edge(node sender,
                                                       node receiver) {
  auto sender_impl = sycl::detail::getSyclObjImpl(sender);
  auto receiver_impl = sycl::detail::getSyclObjImpl(receiver);

  sender_impl->register_successor(receiver_impl); // register successor
  impl->remove_root(receiver_impl); // remove receiver from root node list
}

template <>
command_graph<graph_state::executable>
command_graph<graph_state::modifiable>::finalize(
    const sycl::context &ctx, const sycl::property_list &) const {
  return command_graph<graph_state::executable>{this->impl, ctx};
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
