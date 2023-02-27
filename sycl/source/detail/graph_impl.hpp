//==--------- graph_impl.hpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/handler.hpp>

#include <functional>
#include <list>
#include <set>

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

class wrapper {
  using T = std::function<void(sycl::handler &)>;
  T MFunc;
  std::vector<sycl::event> MDeps;

public:
  wrapper(T t, const std::vector<sycl::event> &deps) : MFunc(t), MDeps(deps){};

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

  std::vector<sycl::detail::ArgDesc> MArgs;

  void exec(sycl::detail::queue_ptr q);

  void register_successor(node_ptr n) {
    MSuccessors.push_back(n);
    n->register_predecessor(node_ptr(this));
  }

  void register_predecessor(node_ptr n) { MPredecessors.push_back(n); }

  sycl::event get_event(void) const { return MEvent; }

  template <typename T>
  node_impl(graph_ptr g, T cgf, const std::vector<sycl::detail::ArgDesc> &args)
      : MScheduled(false), MGraph(g), MBody(cgf), MArgs(args) {
    for (size_t i = 0; i < MArgs.size(); i++) {
      if (MArgs[i].MType == sycl::detail::kernel_param_kind_t::kind_pointer) {
        // Make sure we are storing the actual USM pointer for comparison
        // purposes, note we couldn't actually submit using these copies of the
        // args if subsequent code expects a void**.
        MArgs[i].MPtr = *(void **)(MArgs[i].MPtr);
      }
    }
  }

  // Recursively adding nodes to execution stack:
  void topology_sort(std::list<node_ptr> &schedule) {
    MScheduled = true;
    for (auto i : MSuccessors) {
      if (!i->MScheduled)
        i->topology_sort(schedule);
    }
    schedule.push_front(node_ptr(this));
  }

  bool has_arg(const sycl::detail::ArgDesc &arg, bool dereferencePtr = false) {
    for (auto &nodeArg : MArgs) {
      if (arg.MType == nodeArg.MType && arg.MSize == nodeArg.MSize) {
        // Args coming directly from the handler will need to be dereferenced
        // since they are actually void**
        void *incomingPtr = dereferencePtr ? *(void **)arg.MPtr : arg.MPtr;
        if (incomingPtr == nodeArg.MPtr) {
          return true;
        }
      }
    }
    return false;
  }
};

struct graph_impl {
  std::set<node_ptr> MRoots;
  std::list<node_ptr> MSchedule;
  // TODO: Change one time initialization to per executable object
  bool MFirst;

  graph_ptr MParent;

  void exec(const sycl::detail::queue_ptr &q);
  void exec_and_wait(const sycl::detail::queue_ptr &q);

  void add_root(node_ptr n);
  void remove_root(node_ptr n);

  template <typename T>
  node_ptr add(graph_ptr impl, T cgf,
               const std::vector<sycl::detail::ArgDesc> &args,
               const std::vector<node_ptr> &dep = {});

  graph_impl() : MFirst(true) {}

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  void add_queue(sycl::detail::queue_ptr recordingQueue) {
    MRecordingQueues.insert(recordingQueue);
  }

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  void remove_queue(sycl::detail::queue_ptr recordingQueue) {
    MRecordingQueues.erase(recordingQueue);
  }

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state. \return True if any queues were
  /// removed.
  bool clear_queues();

private:
  std::set<sycl::detail::queue_ptr> MRecordingQueues;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
