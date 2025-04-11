//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>               // for detail::AccessorBaseHost
#include <sycl/context.hpp>                // for context
#include <sycl/detail/export.hpp>          // for __SYCL_EXPORT
#include <sycl/detail/kernel_desc.hpp>     // for kernel_param_kind_t
#include <sycl/detail/owner_less_base.hpp> // for OwnerLessBase
#include <sycl/detail/property_helper.hpp> // for DataLessPropKind, PropWith...
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/string_view.hpp>
#endif
#include <sycl/device.hpp> // for device
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.hpp> // for graph properties classes
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp> // for dynamic_work_group_memory
#include <sycl/ext/oneapi/properties/properties.hpp> // for empty_properties_t
#include <sycl/nd_range.hpp>                         // for range, nd_range
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

/// State to template the command_graph class on.
enum class graph_state {
  modifiable, ///< In modifiable state, commands can be added to graph.
  executable, ///< In executable state, the graph is ready to execute.
};

// Forward declare ext::oneapi::experimental classes
template <graph_state State> class command_graph;
class raw_kernel_arg;
template <typename, typename> class work_group_memory;

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

class node_impl;
class graph_impl;
class exec_graph_impl;
class dynamic_parameter_impl;
class dynamic_command_group_impl;
} // namespace detail

enum class node_type {
  empty = 0,
  subgraph = 1,
  kernel = 2,
  memcpy = 3,
  memset = 4,
  memfill = 5,
  prefetch = 6,
  memadvise = 7,
  ext_oneapi_barrier = 8,
  host_task = 9,
  native_command = 10,
  async_malloc = 11,
  async_free = 12
};

/// Class representing a node in the graph, returned by command_graph::add().
class __SYCL_EXPORT node {
public:
  node() = delete;

  /// Get the type of command associated with this node.
  node_type get_type() const;

  /// Get a list of all the node dependencies of this node.
  std::vector<node> get_predecessors() const;

  /// Get a list of all nodes which depend on this node.
  std::vector<node> get_successors() const;

  /// Get the node associated with a SYCL event returned from a queue recording
  /// submission.
  static node get_node_from_event(event nodeEvent);

  /// Update the ND-Range of this node if it is a kernel execution node
  template <int Dimensions>
  void update_nd_range(nd_range<Dimensions> executionRange);

  /// Update the Range of this node if it is a kernel execution node
  template <int Dimensions> void update_range(range<Dimensions> executionRange);

  /// Common Reference Semantics
  friend bool operator==(const node &LHS, const node &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const node &LHS, const node &RHS) {
    return !operator==(LHS, RHS);
  }

private:
  node(const std::shared_ptr<detail::node_impl> &Impl) : impl(Impl) {}

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  std::shared_ptr<detail::node_impl> impl;
};

namespace property::node {
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
} // namespace property::node

class __SYCL_EXPORT dynamic_command_group {
public:
  dynamic_command_group(
      const command_graph<graph_state::modifiable> &Graph,
      const std::vector<std::function<void(handler &)>> &CGFList);

  size_t get_active_index() const;
  void set_active_index(size_t Index);

  /// Common Reference Semantics
  friend bool operator==(const dynamic_command_group &LHS,
                         const dynamic_command_group &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const dynamic_command_group &LHS,
                         const dynamic_command_group &RHS) {
    return !operator==(LHS, RHS);
  }

private:
  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  std::shared_ptr<detail::dynamic_command_group_impl> impl;
};

namespace detail {
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
  void print_graph(const std::string path, bool verbose = false) const;
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

#ifndef ___INTEL_PREVIEW_BREAKING_CHANGES
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
    modifiable_command_graph::print_graph(const std::string path,
                                          bool verbose) const {
  print_graph(sycl::detail::string_view{path}, verbose);
}
#endif

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

namespace detail {
class __SYCL_EXPORT dynamic_parameter_base {
public:
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  dynamic_parameter_base(size_t ParamSize, const void *Data);
  dynamic_parameter_base();
#else
  dynamic_parameter_base() = default;
#endif

  dynamic_parameter_base(
      const std::shared_ptr<detail::dynamic_parameter_impl> &impl);

  dynamic_parameter_base(
      sycl::ext::oneapi::experimental::command_graph<graph_state::modifiable>
          Graph);

  dynamic_parameter_base(
      sycl::ext::oneapi::experimental::command_graph<graph_state::modifiable>
          Graph,
      size_t ParamSize, const void *Data);

  /// Common Reference Semantics
  friend bool operator==(const dynamic_parameter_base &LHS,
                         const dynamic_parameter_base &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const dynamic_parameter_base &LHS,
                         const dynamic_parameter_base &RHS) {
    return !operator==(LHS, RHS);
  }

protected:
  void updateValue(const void *NewValue, size_t Size);

  // Update a sycl_ext_oneapi_raw_kernel_arg parameter. Size parameter is
  // ignored as it represents sizeof(raw_kernel_arg), which doesn't represent
  // the number of underlying bytes.
  void updateValue(const raw_kernel_arg *NewRawValue, size_t Size);

  void updateAccessor(const sycl::detail::AccessorBaseHost *Acc);

  std::shared_ptr<dynamic_parameter_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
};

class __SYCL_EXPORT dynamic_work_group_memory_base
    : public dynamic_parameter_base {

public:
  dynamic_work_group_memory_base() = default;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  dynamic_work_group_memory_base(size_t BufferSizeInBytes);
#endif
  // TODO: Remove in next ABI breaking window
  dynamic_work_group_memory_base(
      experimental::command_graph<graph_state::modifiable> Graph,
      size_t BufferSizeInBytes);

protected:
  void updateWorkGroupMem(size_t NewBufferSizeInBytes);
};

class __SYCL_EXPORT dynamic_local_accessor_base
    : public dynamic_parameter_base {
public:
  dynamic_local_accessor_base() = default;

  dynamic_local_accessor_base(sycl::range<3> AllocationSize, int Dims,
                              int ElemSize, const property_list &PropList);

protected:
  void updateLocalAccessor(sycl::range<3> NewAllocationSize);
};

} // namespace detail

template <typename DataT, typename PropertyListT = empty_properties_t>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(dynamic_work_group_memory) dynamic_work_group_memory
#ifndef __SYCL_DEVICE_ONLY__
    : public detail::dynamic_work_group_memory_base
#endif
{
public:
  // Check that DataT is an unbounded array type.
  static_assert(std::is_array_v<DataT> && std::extent_v<DataT, 0> == 0);
  static_assert(std::is_same_v<PropertyListT, empty_properties_t>);

  // Frontend requires special types to have a default constructor in order to
  // have a uniform way of initializing an object of special type to then call
  // the __init method on it. This is purely an implementation detail and not
  // part of the spec.
  // TODO: Revisit this once https://github.com/intel/llvm/issues/16061 is
  // closed.
  dynamic_work_group_memory() = default;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Constructs a new dynamic_work_group_memory object.
  /// @param Num Number of elements in the unbounded array DataT.
  dynamic_work_group_memory(size_t Num)
#ifndef __SYCL_DEVICE_ONLY__
      : detail::dynamic_work_group_memory_base(
            Num * sizeof(std::remove_extent_t<DataT>))
#endif
  {
  }
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  __SYCL_DEPRECATED("Dynamic_work_group_memory constructors taking a graph "
                    "object have been deprecated "
                    "and will be removed in the next ABI breaking window.")
#endif
  /// Constructs a new dynamic_work_group_memory object.
  /// @param Graph The graph associated with this object.
  /// @param Num Number of elements in the unbounded array DataT.
  dynamic_work_group_memory(
      [[maybe_unused]] experimental::command_graph<graph_state::modifiable>
          Graph,
      [[maybe_unused]] size_t Num)
#ifndef __SYCL_DEVICE_ONLY__
      : detail::dynamic_work_group_memory_base(
            Graph, Num * sizeof(std::remove_extent_t<DataT>))
#endif
  {
  }

  work_group_memory<DataT, PropertyListT> get() const {
#ifndef __SYCL_DEVICE_ONLY__
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Error: dynamic_work_group_memory::get() can be only "
                          "called on the device!");
#endif
    return WorkGroupMem;
  }

  /// Updates on the host this dynamic_work_group_memory and all registered
  /// nodes with a new buffer size.
  /// @param Num The new number of elements in the unbounded array.
  void update([[maybe_unused]] size_t Num) {
#ifndef __SYCL_DEVICE_ONLY__
    updateWorkGroupMem(Num * sizeof(std::remove_extent_t<DataT>));
#endif
  }

private:
  work_group_memory<DataT, PropertyListT> WorkGroupMem;

#ifdef __SYCL_DEVICE_ONLY__
  using value_type = std::remove_all_extents_t<DataT>;
  using decoratedPtr = typename sycl::detail::DecoratedType<
      value_type, access::address_space::local_space>::type *;

  void __init(decoratedPtr Ptr) { this->WorkGroupMem.__init(Ptr); }
#endif

#ifdef __SYCL_DEVICE_ONLY__
  [[maybe_unused]] unsigned char
      Padding[sizeof(detail::dynamic_work_group_memory_base)];
#endif
};

template <typename DataT, int Dimensions = 1>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(dynamic_local_accessor) dynamic_local_accessor
#ifndef __SYCL_DEVICE_ONLY__
    : public detail::dynamic_local_accessor_base
#endif
{
public:
  static_assert(Dimensions > 0 && Dimensions <= 3);

  // Frontend requires special types to have a default constructor in order to
  // have a uniform way of initializing an object of special type to then call
  // the __init method on it. This is purely an implementation detail and not
  // part of the spec.
  // TODO: Revisit this once https://github.com/intel/llvm/issues/16061 is
  // closed.
  dynamic_local_accessor() = default;

  /// Constructs a new dynamic_local_accessor object.
  /// @param Graph The graph associated with this object.
  /// @param AllocationSize The size of the local accessor.
  /// @param PropList List of properties for the underlying accessor.
  dynamic_local_accessor(
      [[maybe_unused]] experimental::command_graph<graph_state::modifiable>
          Graph,
      [[maybe_unused]] range<Dimensions> AllocationSize,
      [[maybe_unused]] const property_list &PropList = {})
#ifndef __SYCL_DEVICE_ONLY__
      : detail::dynamic_local_accessor_base(
            detail::convertToArrayOfN<3, 1>(AllocationSize), Dimensions,
            sizeof(DataT), PropList)
#endif
  {
  }

  local_accessor<DataT, Dimensions> get() const {
#ifndef __SYCL_DEVICE_ONLY__
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Error: dynamic_local_accessor::get() can be only "
                          "called on the device!");
#endif
    return LocalAccessor;
  }

  /// Updates on the host this dynamic_local_accessor and all registered
  /// nodes with a new size.
  /// @param Num The new number of elements in the unbounded array.
  void update([[maybe_unused]] range<Dimensions> NewAllocationSize) {
#ifndef __SYCL_DEVICE_ONLY__
    updateLocalAccessor(detail::convertToArrayOfN<3, 1>(NewAllocationSize));
#endif
  }

private:
  local_accessor<DataT, Dimensions> LocalAccessor;

#ifdef __SYCL_DEVICE_ONLY__
  void __init(typename local_accessor<DataT, Dimensions>::ConcreteASPtrType Ptr,
              range<Dimensions> AccessRange, range<Dimensions> range,
              id<Dimensions> id) {
    this->LocalAccessor.__init(Ptr, AccessRange, range, id);
  }
#endif

#ifdef __SYCL_DEVICE_ONLY__
  [[maybe_unused]] unsigned char
      Padding[sizeof(detail::dynamic_local_accessor_base)];
#endif
};

template <typename ValueT>
class dynamic_parameter : public detail::dynamic_parameter_base {
  static constexpr bool IsAccessor =
      std::is_base_of_v<sycl::detail::AccessorBaseHost, ValueT>;
  static constexpr sycl::detail::kernel_param_kind_t ParamType =
      IsAccessor ? sycl::detail::kernel_param_kind_t::kind_accessor
      : std::is_pointer_v<ValueT>
          ? sycl::detail::kernel_param_kind_t::kind_pointer
          : sycl::detail::kernel_param_kind_t::kind_std_layout;

public:
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Constructs a new dynamic parameter.
  /// @param Graph The graph associated with this parameter.
  /// @param Param A reference value for this parameter used for CTAD.
  dynamic_parameter(const ValueT &Param)
      : detail::dynamic_parameter_base(sizeof(ValueT), &Param) {}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  __SYCL_DEPRECATED("Dynamic_parameter constructors taking a graph object have "
                    "been deprecated "
                    "and will be removed in the next ABI breaking window.")
#endif
  /// Constructs a new dynamic parameter.
  /// @param Graph The graph associated with this parameter.
  /// @param Param A reference value for this parameter used for CTAD.
  dynamic_parameter(experimental::command_graph<graph_state::modifiable> Graph,
                    const ValueT &Param)
      : detail::dynamic_parameter_base(Graph, sizeof(ValueT), &Param) {}

  /// Updates this dynamic parameter and all registered nodes with a new value.
  /// @param NewValue The new value for the parameter.
  void update(const ValueT &NewValue) {
    if constexpr (IsAccessor) {
      detail::dynamic_parameter_base::updateAccessor(&NewValue);
    } else {
      detail::dynamic_parameter_base::updateValue(&NewValue, sizeof(ValueT));
    }
  }
};

/// Additional CTAD deduction guides.
template <typename ValueT>
dynamic_parameter(experimental::command_graph<graph_state::modifiable> Graph,
                  const ValueT &Param) -> dynamic_parameter<ValueT>;
template <graph_state State = graph_state::modifiable>
command_graph(const context &SyclContext, const device &SyclDevice,
              const property_list &PropList) -> command_graph<State>;

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct __SYCL_EXPORT hash<sycl::ext::oneapi::experimental::node> {
  size_t operator()(const sycl::ext::oneapi::experimental::node &Node) const;
};

template <>
struct __SYCL_EXPORT
    hash<sycl::ext::oneapi::experimental::dynamic_command_group> {
  size_t operator()(const sycl::ext::oneapi::experimental::dynamic_command_group
                        &DynamicCGH) const;
};

template <sycl::ext::oneapi::experimental::graph_state State>
struct hash<sycl::ext::oneapi::experimental::command_graph<State>> {
  size_t operator()(const sycl::ext::oneapi::experimental::command_graph<State>
                        &Graph) const {
    auto ID = sycl::detail::getSyclObjImpl(Graph)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};

template <typename ValueT>
struct hash<sycl::ext::oneapi::experimental::dynamic_parameter<ValueT>> {
  size_t
  operator()(const sycl::ext::oneapi::experimental::dynamic_parameter<ValueT>
                 &DynamicParam) const {
    auto ID = sycl::detail::getSyclObjImpl(DynamicParam)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};

template <typename DataT>
struct hash<sycl::ext::oneapi::experimental::dynamic_work_group_memory<DataT>> {
  size_t operator()(
      const sycl::ext::oneapi::experimental::dynamic_work_group_memory<DataT>
          &DynWorkGroupMem) const {
    auto ID = sycl::detail::getSyclObjImpl(DynWorkGroupMem)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};
} // namespace std
