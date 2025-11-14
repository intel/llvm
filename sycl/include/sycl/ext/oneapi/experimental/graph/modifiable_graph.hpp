//==--------- modifiable_graph.hpp --- SYCL graph extension ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"                                  // for graph_state
#include <sycl/detail/export.hpp>                      // for __SYCL_EXPORT
#include <sycl/detail/owner_less_base.hpp>             // for OwnerLessBase
#include <sycl/detail/string_view.hpp>                 // for string_view
#include <sycl/ext/oneapi/experimental/graph/node.hpp> // for node class
#include <sycl/property_list.hpp>                      // for property_list

#include <functional> // for function
#include <memory>     // for shared_ptr
#include <vector>     // for vector

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

  /// Constructor with default context.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  modifiable_command_graph(const device &SyclDevice,
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
  void print_graph(const std::string path, bool verbose = false) const {
    print_graph(sycl::detail::string_view{path}, verbose);
  }

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

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
