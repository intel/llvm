//==--------- command_graph.hpp --- SYCL graph extension
//---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>          // for __SYCL_EXPORT
#include <sycl/detail/owner_less_base.hpp> // for OwnerLessBase
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/string_view.hpp> // for string_view
#endif
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.hpp> // for graph_state
#include <sycl/ext/oneapi/experimental/graph_node.hpp> // for node class
#include <sycl/property_list.hpp>                      // for property_list

#include <functional>  // for function
#include <memory>      // for shared_ptr
#include <type_traits> // for true_type
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {
// Forward declarations
class handler;
class queue;
class device;
class context;

namespace ext {
namespace oneapi {
namespace experimental {
// Forward declarations
template <graph_state State> class command_graph;
class dynamic_command_group;

namespace detail {
// Forward declarations
class graph_impl;
class exec_graph_impl;

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
  sycl_ext_oneapi_bindless_images = 7,
  sycl_ext_oneapi_experimental_cuda_cluster_launch = 8,
  sycl_ext_codeplay_enqueue_native_command = 9,
  sycl_ext_oneapi_work_group_scratch_memory = 10,
  sycl_ext_oneapi_async_alloc = 11
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
  case UGF::sycl_ext_oneapi_experimental_cuda_cluster_launch:
    return "sycl_ext_oneapi_experimental_cuda_cluster_launch";
  case UGF::sycl_ext_codeplay_enqueue_native_command:
    return "sycl_ext_codeplay_enqueue_native_command";
  case UGF::sycl_ext_oneapi_work_group_scratch_memory:
    return "sycl_ext_oneapi_work_group_scratch_memory";
  case UGF::sycl_ext_oneapi_async_alloc:
    return "sycl_ext_oneapi_async_alloc";
  }

  assert(false && "Unhandled graphs feature");
  return {};
}

// Templateless modifiable command-graph base class.
class __SYCL_EXPORT modifiable_command_graph
    : public sycl::detail::OwnerLessBase<modifiable_command_graph> {
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
    checkNodePropertiesAndThrow(PropList);
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
    checkNodePropertiesAndThrow(PropList);
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
  /// @param PropList Property list used to pass properties for recording.
  void begin_recording(queue &RecordingQueue,
                       const property_list &PropList = {});

  /// Change the state of multiple queues to be recording and associate this
  /// graph with each of them.
  /// @param RecordingQueues The queues to change state on and associate this
  /// graph with.
  /// @param PropList Property list used to pass properties for recording.
  void begin_recording(const std::vector<queue> &RecordingQueues,
                       const property_list &PropList = {});

  /// Set all queues currently recording to this graph to the executing state.
  void end_recording();

  /// Set a queue currently recording to this graph to the executing state.
  /// @param RecordingQueue The queue to change state on.
  void end_recording(queue &RecordingQueue);

  /// Set multiple queues currently recording to this graph to the executing
  /// state.
  /// @param RecordingQueues The queues to change state on.
  void end_recording(const std::vector<queue> &RecordingQueues);

  /// Synchronous operation that writes a DOT formatted description of the graph
  /// to the provided path. By default, this includes the graph topology, node
  /// types, node id and kernel names.
  /// @param path The path to write the DOT file to.
  /// @param verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
#ifdef ___INTEL_PREVIEW_BREAKING_CHANGES
  void print_graph(const std::string path, bool verbose = false) const {
    print_graph(sycl::detail::string_view{path}, verbose);
  }
#else
#ifdef __SYCL_GRAPH_IMPL_CPP
  // Magic combination found by trial and error:
  __SYCL_EXPORT
#if _WIN32
  inline
#endif
#else
  inline
#endif
      void
      print_graph(const std::string path, bool verbose = false) const {
    print_graph(sycl::detail::string_view{path}, verbose);
  }
#endif

  /// Get a list of all nodes contained in this graph.
  std::vector<node> get_nodes() const;

  /// Get a list of all root nodes (nodes without dependencies) in this graph.
  std::vector<node> get_root_nodes() const;

  /// Common Reference Semantics
  friend bool operator==(const modifiable_command_graph &LHS,
                         const modifiable_command_graph &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const modifiable_command_graph &LHS,
                         const modifiable_command_graph &RHS) {
    return !operator==(LHS, RHS);
  }

protected:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  modifiable_command_graph(const std::shared_ptr<detail::graph_impl> &Impl)
      : impl(Impl) {}

  /// Template-less implementation of add() for dynamic command-group nodes.
  /// @param DynCGF Dynamic Command-group function object to add.
  /// @param Dep List of predecessor nodes.
  /// @return Node added to the graph.
  node addImpl(dynamic_command_group &DynCGF, const std::vector<node> &Dep);

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

  void print_graph(sycl::detail::string_view path, bool verbose = false) const;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
  std::shared_ptr<detail::graph_impl> impl;

  static void checkNodePropertiesAndThrow(const property_list &Properties);
};

// Templateless executable command-graph base class.
class __SYCL_EXPORT executable_command_graph
    : public sycl::detail::OwnerLessBase<executable_command_graph> {
public:
  /// An executable command-graph is not user constructable.
  executable_command_graph() = delete;

  /// Update the inputs & output of the graph.
  /// @param Graph Graph to use the inputs and outputs of.
  void update(const command_graph<graph_state::modifiable> &Graph);

  /// Updates a single node in this graph based on the contents of the provided
  /// node.
  /// @param Node The node to use for updating the graph.
  void update(const node &Node);

  /// Updates a number of nodes in this graph based on the contents of the
  /// provided nodes.
  /// @param Nodes The nodes to use for updating the graph.
  void update(const std::vector<node> &Nodes);

  /// Return the total amount of memory required by this graph for graph-owned
  /// memory allocations.
  size_t get_required_mem_size() const;

  /// Common Reference Semantics
  friend bool operator==(const executable_command_graph &LHS,
                         const executable_command_graph &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const executable_command_graph &LHS,
                         const executable_command_graph &RHS) {
    return !operator==(LHS, RHS);
  }

protected:
  /// Constructor used by internal runtime.
  /// @param Graph Detail implementation class to construct with.
  /// @param Ctx Context to use for graph.
  /// @param PropList Optional list of properties to pass.
  executable_command_graph(const std::shared_ptr<detail::graph_impl> &Graph,
                           const sycl::context &Ctx,
                           const property_list &PropList = {});

  template <class Obj>
  friend const decltype(Obj::impl) &
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
  explicit command_graph(const queue &SyclQueue,
                         const property_list &PropList = {})
      : modifiable_command_graph(SyclQueue, PropList) {}

private:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  command_graph(const std::shared_ptr<detail::graph_impl> &Impl)
      : modifiable_command_graph(Impl) {}

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
};

template <>
class command_graph<graph_state::executable>
    : public detail::executable_command_graph {
protected:
  friend command_graph<graph_state::executable>
  detail::modifiable_command_graph::finalize(const sycl::property_list &) const;
  using detail::executable_command_graph::executable_command_graph;
};

/// Additional CTAD deduction guides.
template <graph_state State = graph_state::modifiable>
command_graph(const context &SyclContext, const device &SyclDevice,
              const property_list &PropList) -> command_graph<State>;

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl

namespace std {
template <sycl::ext::oneapi::experimental::graph_state State>
struct hash<sycl::ext::oneapi::experimental::command_graph<State>> {
  size_t operator()(const sycl::ext::oneapi::experimental::command_graph<State>
                        &Graph) const {
    auto ID = sycl::detail::getSyclObjImpl(Graph)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};
} // namespace std
