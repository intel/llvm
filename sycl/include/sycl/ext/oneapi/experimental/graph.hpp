//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <set>
#include <list>

__SYCL_INLINE_NAMESPACE(cl) {
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
    using T = std::function<void(sycl::handler&)>;
    T my_func;
    std::vector<sycl::event> my_deps;
public:
    wrapper(T t, const std::vector<sycl::event>& deps) : my_func(t), my_deps(deps) {};

    void operator()(sycl::handler& cgh) {
        cgh.depends_on(my_deps);
        std::invoke(my_func,cgh);
    }
};

struct node_impl {
    bool is_scheduled;

    graph_ptr my_graph;
    sycl::event my_event;

    std::vector<node_ptr> my_successors;
    std::vector<node_ptr> my_predecessors;

    std::function<void(sycl::handler&)> my_body;

    void exec( sycl::queue q ) {
        std::vector<sycl::event> __deps;
        for(auto i:my_predecessors) __deps.push_back(i->get_event());
        my_event = q.submit(wrapper{my_body,__deps});
    }

    void register_successor(node_ptr n) {
        my_successors.push_back(n);
        n->register_predecessor(node_ptr(this));
    }

    void register_predecessor(node_ptr n) { my_predecessors.push_back(n); }

    sycl::event get_event(void) {return my_event;}

    template<typename T>
    node_impl(graph_ptr g, T cgf) : is_scheduled(false), my_graph(g), my_body(cgf) {}

    // Recursively adding nodes to execution stack:
    void topology_sort(std::list<node_ptr>& schedule) {
        is_scheduled = true;
        for(auto i:my_successors) {
            if(!i->is_scheduled) i->topology_sort(schedule);
        }
        schedule.push_front(node_ptr(this));
    }
};

struct graph_impl {
    std::set<node_ptr> my_roots;
    std::list<node_ptr> my_schedule;

    graph_ptr parent;

    void exec( sycl::queue q ) {
        if( my_schedule.empty() ) {
            for(auto n : my_roots) {
                n->topology_sort(my_schedule);
            }
        }
        for(auto n : my_schedule) n->exec(q);
    }

    void exec_and_wait( sycl::queue q ) {
        exec(q);
        q.wait();
    }

    void add_root(node_ptr n) {
        my_roots.insert(n);
        for(auto n : my_schedule) n->is_scheduled=false;
        my_schedule.clear();
    }

    void remove_root(node_ptr n) {
        my_roots.erase(n);
        for(auto n : my_schedule) n->is_scheduled=false;
        my_schedule.clear();
    }

    graph_impl() {}
};

} // namespace detail

class node;

class graph;

class executable_graph;

struct node {
    // TODO: add properties to distinguish between empty, host, device nodes.
    detail::node_ptr my_node;
    detail::graph_ptr my_graph;

    template<typename T>
    node(detail::graph_ptr g, T cgf) : my_graph(g), my_node(new detail::node_impl(g,cgf)) {};
    void register_successor(node n) { my_node->register_successor(n.my_node); }
    void exec( sycl::queue q, sycl::event = sycl::event() ) { my_node->exec(q); }

    void set_root() {  my_graph->add_root(my_node);}

    // TODO: Add query functions: is_root, ...
};

class executable_graph {
public:
    int my_tag;
    sycl::queue my_queue;

    void exec_and_wait();// { my_queue.wait(); }

    executable_graph(detail::graph_ptr g, sycl::queue q) : my_queue(q), my_tag(rand()) {
      g->exec(my_queue);
    }
};

class graph {
public:
    // Adding empty node with [0..n] predecessors:
    node add_empty_node(const std::vector<node>& dep = {});

    // Adding node for host task 
    template<typename T>
    node add_host_node(T hostTaskCallable, const std::vector<node>& dep = {});

    // Adding device node:
    template<typename T>
    node add_device_node(T cgf, const std::vector<node>& dep = {});

    // Adding dependency between two nodes.
    void make_edge(node sender, node receiver);

    // TODO: Extend queue to directly submit graph
    void exec_and_wait( sycl::queue q );

    executable_graph exec( sycl::queue q ) { return executable_graph{my_graph,q};};

    graph() : my_graph(new detail::graph_impl()) {}

    // Creating a subgraph (with predecessors)
    graph(graph& parent, const std::vector<node>& dep = {}) {}

    bool is_subgraph();

private:
    detail::graph_ptr my_graph;
};

void executable_graph::exec_and_wait() { my_queue.wait(); }

template<typename T>
node graph::add_device_node(T cgf , const std::vector<node>& dep) {
    node _node(my_graph,cgf);
    if( !dep.empty() ) {
        for(auto n : dep) this->make_edge(n,_node);
    } else {
        _node.set_root();
    }
    return _node;
}

void graph::make_edge(node sender, node receiver) {
    sender.register_successor(receiver);//register successor
    my_graph->remove_root(receiver.my_node); //remove receiver from root node list
}

void graph::exec_and_wait( sycl::queue q ) {
    my_graph->exec_and_wait(q);
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

