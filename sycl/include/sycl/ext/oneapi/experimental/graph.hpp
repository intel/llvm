//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <list>
#include <set>

//__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
  T my_func;
  std::vector<sycl::event> my_deps;

public:
  wrapper(T t, const std::vector<sycl::event> &deps)
      : my_func(t), my_deps(deps){};

  void operator()(sycl::handler &cgh) {
    cgh.depends_on(my_deps);
    std::invoke(my_func, cgh);
  }
};

struct node_impl {
  bool is_scheduled;

  graph_ptr my_graph;
  sycl::event my_event;

  std::vector<node_ptr> my_successors;
  std::vector<node_ptr> my_predecessors;

  std::function<void(sycl::handler &)> my_body;

  void exec(sycl::queue q) {
    std::vector<sycl::event> __deps;
    for (auto i : my_predecessors)
      __deps.push_back(i->get_event());
    my_event = q.submit(wrapper{my_body, __deps});
  }

  void register_successor(node_ptr n) {
    my_successors.push_back(n);
    n->register_predecessor(node_ptr(this));
  }

  void register_predecessor(node_ptr n) { my_predecessors.push_back(n); }

  sycl::event get_event(void) { return my_event; }

  template <typename T>
  node_impl(graph_ptr g, T cgf)
      : is_scheduled(false), my_graph(g), my_body(cgf) {}

  // Recursively adding nodes to execution stack:
  void topology_sort(std::list<node_ptr> &schedule) {
    is_scheduled = true;
    for (auto i : my_successors) {
      if (!i->is_scheduled)
        i->topology_sort(schedule);
    }
    schedule.push_front(node_ptr(this));
  }
};

struct graph_impl {
  std::set<node_ptr> my_roots;
  std::list<node_ptr> my_schedule;
  // TODO: Change one time initialization to per executable object
  bool first;

  graph_ptr parent;

  void exec(sycl::queue q) {
    if (my_schedule.empty()) {
      for (auto n : my_roots) {
        n->topology_sort(my_schedule);
      }
    }
    for (auto n : my_schedule)
      n->exec(q);
  }

  void exec_and_wait(sycl::queue q) {
    if (first) {
      exec(q);
      first = false;
    }
    q.wait();
  }

  void add_root(node_ptr n) {
    my_roots.insert(n);
    for (auto n : my_schedule)
      n->is_scheduled = false;
    my_schedule.clear();
  }

  void remove_root(node_ptr n) {
    my_roots.erase(n);
    for (auto n : my_schedule)
      n->is_scheduled = false;
    my_schedule.clear();
  }

  graph_impl() : first(true) {}
};

} // namespace detail

struct node {
  detail::node_ptr my_node;
  detail::graph_ptr my_graph;

  template <typename T>
  node(detail::graph_ptr g, T cgf)
      : my_graph(g), my_node(new detail::node_impl(g, cgf)){};
  void register_successor(node n) { my_node->register_successor(n.my_node); }
  void exec(sycl::queue q, sycl::event = sycl::event()) { my_node->exec(q); }

  void set_root() { my_graph->add_root(my_node); }
};

enum class graph_state { modifiable, executable };

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

  command_graph() : my_graph(new detail::graph_impl()) {}

private:
  detail::graph_ptr my_graph;
};

template <> class command_graph<graph_state::executable> {
public:
  int my_tag;
  const sycl::context &my_ctx;

  void exec_and_wait(sycl::queue q);

  command_graph() = delete;

  command_graph(detail::graph_ptr g, const sycl::context &ctx)
      : my_graph(g), my_ctx(ctx), my_tag(rand()) {}

private:
  detail::graph_ptr my_graph;
};

template <>
template <typename T>
node command_graph<graph_state::modifiable>::add(T cgf,
                                                 const std::vector<node> &dep) {
  node _node(my_graph, cgf);
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, _node);
  } else {
    _node.set_root();
  }
  return _node;
}

template <>
void command_graph<graph_state::modifiable>::make_edge(node sender,
                                                       node receiver) {
  sender.register_successor(receiver);     // register successor
  my_graph->remove_root(receiver.my_node); // remove receiver from root node
                                           // list
}

template <>
command_graph<graph_state::executable>
command_graph<graph_state::modifiable>::finalize(
    const sycl::context &ctx) const {
  return command_graph<graph_state::executable>{this->my_graph, ctx};
}

void command_graph<graph_state::executable>::exec_and_wait(sycl::queue q) {
  my_graph->exec_and_wait(q);
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
//} // __SYCL_INLINE_NAMESPACE(cl)
