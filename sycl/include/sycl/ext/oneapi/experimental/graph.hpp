//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>                // for context
#include <sycl/detail/export.hpp>          // for __SYCL_EXPORT
#include <sycl/detail/property_helper.hpp> // for DataLessPropKind, PropWith...
#include <sycl/device.hpp>                 // for device
#include <sycl/properties/property_traits.hpp> // for is_property, is_property_of
#include <sycl/property_list.hpp>              // for property_list

#include <functional>  // for function
#include <memory>      // for shared_ptr
#include <type_traits> // for true_type
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {

class handler;
class queue;
class device;
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
// List of sycl features and extensions which are not supported by graphs. Used
// for throwing errors when these features are used with graphs.
enum class UnsupportedGraphFeatures {
  sycl_reductions = 0,
  sycl_specialization_constants = 1,
  sycl_kernel_bundle = 2,
  sycl_ext_oneapi_kernel_properties = 3,
  sycl_ext_oneapi_enqueue_barrier = 4,
  sycl_ext_oneapi_memcpy2d = 5,
  sycl_ext_oneapi_device_global = 6,
  sycl_ext_oneapi_bindless_images = 7
};

inline const char *
UnsupportedFeatureToString(UnsupportedGraphFeatures Feature) {
  using UGF = UnsupportedGraphFeatures;
  switch (Feature) {
  case UGF::sycl_reductions:
    return "Reductions";
  case UGF::sycl_specialization_constants:
    return "Specialization Constants";
  case UGF::sycl_kernel_bundle:
    return "Kernel Bundles";
  case UGF::sycl_ext_oneapi_kernel_properties:
    return "sycl_ext_oneapi_kernel_properties";
  case UGF::sycl_ext_oneapi_enqueue_barrier:
    return "sycl_ext_oneapi_enqueue_barrier";
  case UGF::sycl_ext_oneapi_memcpy2d:
    return "sycl_ext_oneapi_memcpy2d";
  case UGF::sycl_ext_oneapi_device_global:
    return "sycl_ext_oneapi_device_global";
  case UGF::sycl_ext_oneapi_bindless_images:
    return "sycl_ext_oneapi_bindless_images";
  }

  assert(false && "Unhandled graphs feature");
  return {};
}

class node_impl;
class graph_impl;
class exec_graph_impl;

} // namespace detail

/// State to template the command_graph class on.
enum class graph_state {
  modifiable, ///< In modifiable state, commands can be added to graph.
  executable, ///< In executable state, the graph is ready to execute.
};

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
};

namespace property {
namespace graph {

/// Property passed to command_graph constructor to disable checking for cycles.
///
class no_cycle_check : public ::sycl::detail::DataLessProperty<
                           ::sycl::detail::GraphNoCycleCheck> {
public:
  no_cycle_check() = default;
};

/// Property passed to command_graph constructor to allow buffers to be used
/// with graphs. Passing this property represents a promise from the user that
/// the buffer will outlive any graph that it is used in.
///
class assume_buffer_outlives_graph
    : public ::sycl::detail::DataLessProperty<
          ::sycl::detail::GraphAssumeBufferOutlivesGraph> {
public:
  assume_buffer_outlives_graph() = default;
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

/// Property used to to add all previous graph leaves as dependencies when
/// creating a new node with command_graph::add().
class depends_on_all_leaves : public ::sycl::detail::DataLessProperty<
                                  ::sycl::detail::GraphDependOnAllLeaves> {
public:
  depends_on_all_leaves() = default;
};

} // namespace node
} // namespace property

template <graph_state State> class command_graph;

namespace detail {
// Templateless modifiable command-graph base class.
class __SYCL_EXPORT modifiable_command_graph {
public:
  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  modifiable_command_graph(const context &SyclContext, const device &SyclDevice,
                           const property_list &PropList = {});

  /// Constructor.
  /// @param SyclQueue Queue to use for the graph device and context.
  /// @param PropList Optional list of properties to pass.
  modifiable_command_graph(const queue &SyclQueue,
                           const property_list &PropList = {});

  /// Add an empty node to the graph.
  /// @param PropList Property list used to pass [0..n] predecessor nodes.
  /// @return Constructed empty node which has been added to the graph.
  node add(const property_list &PropList = {}) {
    if (PropList.has_property<property::node::depends_on>()) {
      auto Deps = PropList.get_property<property::node::depends_on>();
      node Node = addImpl(Deps.get_dependencies());
      if (PropList.has_property<property::node::depends_on_all_leaves>()) {
        addGraphLeafDependencies(Node);
      }
      return Node;
    }
    node Node = addImpl({});
    if (PropList.has_property<property::node::depends_on_all_leaves>()) {
      addGraphLeafDependencies(Node);
    }
    return Node;
  }

  /// Add a command-group node to the graph.
  /// @param CGF Command-group function to create node with.
  /// @param PropList Property list used to pass [0..n] predecessor nodes.
  /// @return Constructed node which has been added to the graph.
  template <typename T> node add(T CGF, const property_list &PropList = {}) {
    if (PropList.has_property<property::node::depends_on>()) {
      auto Deps = PropList.get_property<property::node::depends_on>();
      node Node = addImpl(CGF, Deps.get_dependencies());
      if (PropList.has_property<property::node::depends_on_all_leaves>()) {
        addGraphLeafDependencies(Node);
      }
      return Node;
    }
    node Node = addImpl(CGF, {});
    if (PropList.has_property<property::node::depends_on_all_leaves>()) {
      addGraphLeafDependencies(Node);
    }
    return Node;
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

  /// Set a queue currently recording to this graph to the executing state.
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

  /// Synchronous operation that writes a DOT formatted description of the graph
  /// to the provided path. By default, this includes the graph topology, node
  /// types, node id and kernel names.
  /// @param path The path to write the DOT file to.
  /// @param verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void print_graph(const std::string path, bool verbose = false) const;

protected:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  modifiable_command_graph(const std::shared_ptr<detail::graph_impl> &Impl)
      : impl(Impl) {}

  /// Template-less implementation of add() for CGF nodes.
  /// @param CGF Command-group function to add.
  /// @param Dep List of predecessor nodes.
  /// @return Node added to the graph.
  node addImpl(std::function<void(handler &)> CGF,
               const std::vector<node> &Dep);

  /// Template-less implementation of add() for empty nodes.
  /// @param Dep List of predecessor nodes.
  /// @return Node added to the graph.
  node addImpl(const std::vector<node> &Dep);

  /// Adds all graph leaves as dependencies
  /// @param Node Destination node to which the leaves of the graph will be
  /// added as dependencies.
  void addGraphLeafDependencies(node Node);

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  std::shared_ptr<detail::graph_impl> impl;
};

// Templateless executable command-graph base class.
class __SYCL_EXPORT executable_command_graph {
public:
  /// An executable command-graph is not user constructable.
  executable_command_graph() = delete;

  /// Update the inputs & output of the graph.
  /// @param Graph Graph to use the inputs and outputs of.
  void update(const command_graph<graph_state::modifiable> &Graph);

protected:
  /// Constructor used by internal runtime.
  /// @param Graph Detail implementation class to construct with.
  /// @param Ctx Context to use for graph.
  executable_command_graph(const std::shared_ptr<detail::graph_impl> &Graph,
                           const sycl::context &Ctx);

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  /// Creates a backend representation of the graph in \p impl member variable.
  void finalizeImpl();

  std::shared_ptr<detail::exec_graph_impl> impl;
};
} // namespace detail

/// Graph in the modifiable state.
template <graph_state State = graph_state::modifiable>
class command_graph : public detail::modifiable_command_graph {
public:
  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  command_graph(const context &SyclContext, const device &SyclDevice,
                const property_list &PropList = {})
      : modifiable_command_graph(SyclContext, SyclDevice, PropList) {}

  /// Constructor.
  /// @param SyclQueue Queue to use for the graph device and context.
  /// @param PropList Optional list of properties to pass.
  command_graph(const queue &SyclQueue, const property_list &PropList = {})
      : modifiable_command_graph(SyclQueue, PropList) {}

private:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  command_graph(const std::shared_ptr<detail::graph_impl> &Impl)
      : modifiable_command_graph(Impl) {}
};

template <>
class command_graph<graph_state::executable>
    : public detail::executable_command_graph {

protected:
  friend command_graph<graph_state::executable>
  detail::modifiable_command_graph::finalize(const sycl::property_list &) const;
  using detail::executable_command_graph::executable_command_graph;
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

} // namespace _V1
} // namespace sycl
