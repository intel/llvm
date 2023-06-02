//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/property_list.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;
class queue;
class device;
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
class node_impl;
class graph_impl;
class exec_graph_impl;

} // namespace detail

/// State to template the command_graph class on.
enum class graph_state {
  modifiable, ///< In modifiable state, commands can be added to graph.
  executable, ///< In executable state, the graph is ready to execute.
};

// Forward declaration
class node;

namespace property {
namespace graph {

/// Property passed to command_graph constructor to disable checking for cycles.
///
/// \todo Cycle check not yet implemented.
class no_cycle_check : public ::sycl::detail::DataLessProperty<
                           ::sycl::detail::GraphNoCycleCheck> {
public:
  no_cycle_check() = default;
};

} // namespace graph

namespace node {

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

} // namespace node
} // namespace property

/// Class representing a node in the graph, returned by command_graph::add().
class __SYCL_EXPORT node {
private:
  node(const std::shared_ptr<detail::node_impl> &Impl) : impl(Impl) {}

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  std::shared_ptr<detail::node_impl> impl;
  std::shared_ptr<detail::graph_impl> MGraph;
};

/// Class representing a graph in the modifiable state.
template <graph_state State = graph_state::modifiable>
class __SYCL_EXPORT command_graph {
public:
  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  command_graph(const context &SyclContext, const device &SyclDevice,
                const property_list &PropList = {});

  /// Add an empty node to the graph.
  /// @param PropList Property list used to pass [0..n] predecessor nodes.
  /// @return Constructed empty node which has been added to the graph.
  node add(const property_list &PropList = {}) {
    if (PropList.has_property<property::node::depends_on>()) {
      auto Deps = PropList.get_property<property::node::depends_on>();
      return add_impl(Deps.get_dependencies());
    }
    return add_impl({});
  }

  /// Add a command-group node to the graph.
  /// @param CGF Command-group function to create node with.
  /// @param PropList Property list used to pass [0..n] predecessor nodes.
  /// @return Constructed node which has been added to the graph.
  template <typename T> node add(T CGF, const property_list &PropList = {}) {
    if (PropList.has_property<property::node::depends_on>()) {
      auto Deps = PropList.get_property<property::node::depends_on>();
      return add_impl(CGF, Deps.get_dependencies());
    }
    return add_impl(CGF, {});
  }

  /// Add a dependency between two nodes.
  /// @param Src Node which will be a dependency of \p Dest.
  /// @param Dest Node which will be dependent on \p Src.
  void make_edge(node &Src, node &Dest);

  /// Finalize modifiable graph into an executable graph.
  /// @param PropList Property list used to pass properties for finalization.
  /// @return Executable graph object.
  command_graph<graph_state::executable>
  finalize(const property_list &PropList = {}) const;

  /// Change the state of a queue to be recording and associate this graph with
  /// it.
  /// @param RecordingQueue The queue to change state on and associate this
  /// graph with.
  /// @return True if the queue had its state changed from executing to
  /// recording.
  bool begin_recording(queue &RecordingQueue);

  /// Change the state of multiple queues to be recording and associate this
  /// graph with each of them.
  /// @param RecordingQueues The queues to change state on and associate this
  /// graph with.
  /// @return True if any queue had its state changed from executing to
  /// recording.
  bool begin_recording(const std::vector<queue> &RecordingQueues);

  /// Set all queues currently recording to this graph to the executing state.
  /// @return True if any queue had its state changed from recording to
  /// executing.
  bool end_recording();

  /// Set a queues currently recording to this graph to the executing state.
  /// @param RecordingQueue The queue to change state on.
  /// @return True if the queue had its state changed from recording to
  /// executing.
  bool end_recording(queue &RecordingQueue);

  /// Set multiple queues currently recording to this graph to the executing
  /// state.
  /// @param RecordingQueues The queues to change state on.
  /// @return True if any queue had its state changed from recording to
  /// executing.
  bool end_recording(const std::vector<queue> &RecordingQueues);

private:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  command_graph(const std::shared_ptr<detail::graph_impl> &Impl) : impl(Impl) {}

  /// Template-less implementation of add() for CGF nodes.
  /// @param CGF Command-group function to add.
  /// @param Dep List of predecessor nodes.
  /// @return Node added to the graph.
  node add_impl(std::function<void(handler &)> CGF,
                const std::vector<node> &Dep);

  /// Template-less implementation of add() for empty nodes.
  /// @param Dep List of predecessor nodes.
  /// @return Node added to the graph.
  node add_impl(const std::vector<node> &Dep);

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  std::shared_ptr<detail::graph_impl> impl;
};

template <> class __SYCL_EXPORT command_graph<graph_state::executable> {
public:
  /// An executable command-graph is not user constructable.
  command_graph() = delete;

  /// Constructor used by internal runtime.
  /// @param Graph Detail implementation class to construct with.
  /// @param Ctx Context to use for graph.
  command_graph(const std::shared_ptr<detail::graph_impl> &Graph,
                const sycl::context &Ctx);

  /// Update the inputs & output of the graph.
  /// @param Graph Graph to use the inputs and outputs of.
  void update(const command_graph<graph_state::modifiable> &Graph);

private:
  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  /// Creates a backend representation of the graph in \p impl member variable.
  void finalize_impl();

  int MTag;
  std::shared_ptr<detail::exec_graph_impl> impl;
};

/// Additional CTAD deduction guide.
template <graph_state State = graph_state::modifiable>
command_graph(const context &SyclContext, const device &SyclDevice,
              const property_list &PropList) -> command_graph<State>;

} // namespace experimental
} // namespace oneapi
} // namespace ext

template <>
struct is_property<ext::oneapi::experimental::property::graph::no_cycle_check>
    : std::true_type {};

template <>
struct is_property<ext::oneapi::experimental::property::node::depends_on>
    : std::true_type {};

template <>
struct is_property_of<
    ext::oneapi::experimental::property::graph::no_cycle_check,
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>> : std::true_type {
};

template <>
struct is_property_of<ext::oneapi::experimental::property::node::depends_on,
                      ext::oneapi::experimental::node> : std::true_type {};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
