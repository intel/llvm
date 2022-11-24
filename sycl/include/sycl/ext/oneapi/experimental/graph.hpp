//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/queue.hpp>

#include "graph_defines.hpp"

#include <list>
#include <set>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
class queue;

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

struct node_impl;

struct graph_impl;

using node_ptr = std::shared_ptr<node_impl>;

using graph_ptr = std::shared_ptr<graph_impl>;

class wrapper {
  using T = std::function<void(sycl::handler &)>;
  T MFunc;
  std::vector<sycl::event> MDeps;

public:
  wrapper(T t, const std::vector<sycl::event> &deps)
      : MFunc(t), MDeps(deps){};

  void operator()(sycl::handler &cgh) {
    cgh.depends_on(MDeps);
    std::invoke(MFunc, cgh);
  }
};

struct node_impl {
  bool MScheduled;

  graph_ptr MGraph;
  sycl::event MEvent;

  std::vector<node_ptr> MSuccessors;
  std::vector<node_ptr> MPredecessors;

  std::function<void(sycl::handler &)> MBody;

  void exec(sycl::queue q) {
    std::vector<sycl::event> deps;
    for (auto i : MPredecessors)
      deps.push_back(i->get_event());
    MEvent = q.submit(wrapper{MBody, deps});
  }

  void register_successor(node_ptr n) {
    MSuccessors.push_back(n);
    n->register_predecessor(node_ptr(this));
  }

  void register_predecessor(node_ptr n) { MPredecessors.push_back(n); }

  sycl::event get_event(void) { return MEvent; }

  template <typename T>
  node_impl(graph_ptr g, T cgf)
      : MScheduled(false), MGraph(g), MBody(cgf) {}

  // Recursively adding nodes to execution stack:
  void topology_sort(std::list<node_ptr> &schedule) {
    MScheduled = true;
    for (auto i : MSuccessors) {
      if (!i->MScheduled)
        i->topology_sort(schedule);
    }
    schedule.push_front(node_ptr(this));
  }
};

struct graph_impl {
  std::set<node_ptr> MRoots;
  std::list<node_ptr> MSchedule;
  // TODO: Change one time initialization to per executable object
  bool MFirst;

  graph_ptr MParent;

  void exec(sycl::queue q) {
    if (MSchedule.empty()) {
      for (auto n : MRoots) {
        n->topology_sort(MSchedule);
      }
    }
    for (auto n : MSchedule)
      n->exec(q);
  }

  void exec_and_wait(sycl::queue q) {
    if (MFirst) {
      exec(q);
      MFirst = false;
    }
    q.wait();
  }

  void add_root(node_ptr n) {
    MRoots.insert(n);
    for (auto n : MSchedule)
      n->MScheduled = false;
    MSchedule.clear();
  }

  void remove_root(node_ptr n) {
    MRoots.erase(n);
    for (auto n : MSchedule)
      n->MScheduled = false;
    MSchedule.clear();
  }

  template <typename T>
  node_ptr add(graph_ptr impl, T cgf, const std::vector<node_ptr> &dep = {}) {
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

  graph_impl() : MFirst(true) {}
};

} // namespace detail

class node {
public:
  template <typename T>
  node(detail::graph_ptr g, T cgf)
      : MGraph(g), impl(new detail::node_impl(g, cgf)) {}
  void register_successor(node n) { impl->register_successor(n.impl); }
  void exec(sycl::queue q) { impl->exec(q); }

private:
  node(detail::node_ptr Impl) : impl(Impl) {}

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  detail::node_ptr impl;
  detail::graph_ptr MGraph;
};

template <graph_state State = graph_state::modifiable> class command_graph {
public:
  // Adding empty node with [0..n] predecessors:
  node add(const std::vector<node> &dep = {});

  // Adding device node:
  template <typename T> node add(T cgf, const std::vector<node> &dep = {});

  // Adding dependency between two nodes.
  void make_edge(node sender, node receiver);

  // TODO: Extend queue to directly submit graph
  void exec_and_wait(sycl::queue q);

  command_graph<graph_state::executable>
  finalize(const sycl::context &syclContext) const;

  command_graph() : impl(new detail::graph_impl()) {}

private:
  command_graph(detail::graph_ptr Impl) : impl(Impl) {}

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  detail::graph_ptr impl;
};

template <> class command_graph<graph_state::executable> {
public:
  void exec_and_wait(sycl::queue q);

  command_graph() = delete;

  command_graph(detail::graph_ptr g, const sycl::context &ctx)
      : MTag(rand()), MCtx(ctx), impl(g) {}

private:
  int MTag;
  const sycl::context &MCtx;
  detail::graph_ptr impl;
};

template <>
template <typename T>
inline node
command_graph<graph_state::modifiable>::add(T cgf,
                                            const std::vector<node> &dep) {
  std::vector<detail::node_ptr> depImpls;
  for (auto &d : dep) {
    depImpls.push_back(sycl::detail::getSyclObjImpl(d));
  }

  auto nodeImpl = impl->add(impl, cgf, depImpls);
  return sycl::detail::createSyclObjFromImpl<node>(nodeImpl);
}

template <>
inline void command_graph<graph_state::modifiable>::make_edge(node sender,
                                                              node receiver) {
  sender.register_successor(receiver);     // register successor
  impl->remove_root(
      sycl::detail::getSyclObjImpl(receiver)); // remove receiver from root node
                                               // list
}

template <>
command_graph<graph_state::executable> inline command_graph<
    graph_state::modifiable>::finalize(const sycl::context &ctx) const {
  return command_graph<graph_state::executable>{this->impl, ctx};
}

inline void
command_graph<graph_state::executable>::exec_and_wait(sycl::queue q) {
  impl->exec_and_wait(q);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

