//==--------- graph_impl.hpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/handler.hpp>

#include <detail/kernel_impl.hpp>

#include <cstring>
#include <functional>
#include <list>
#include <set>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

struct node_impl {
  // List of successors to this node.
  std::vector<std::shared_ptr<node_impl>> MSuccessors;
  // List of predecessors to this node. Using weak_ptr here to prevent circular
  // references between nodes.
  std::vector<std::weak_ptr<node_impl>> MPredecessors;

  /// Kernel to be executed by this node
  std::shared_ptr<sycl::detail::kernel_impl> MKernel;
  /// Description of the kernel global and local sizes as well as offset
  sycl::detail::NDRDescT MNDRDesc;
  /// Module handle for the kernel to be executed.
  sycl::detail::OSModuleHandle MOSModuleHandle =
      sycl::detail::OSUtil::ExeModuleHandle;
  /// Kernel name inside the module
  std::string MKernelName;
  std::vector<sycl::detail::AccessorImplPtr> MAccStorage;
  std::vector<sycl::detail::LocalAccessorImplPtr> MLocalAccStorage;
  std::vector<std::shared_ptr<sycl::detail::stream_impl>> MStreamStorage;
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;
  sycl::detail::CG::CGTYPE MCGType = sycl::detail::CG::None;

  /// Store arg descriptors for the kernel arguments
  std::vector<sycl::detail::ArgDesc> MArgs;
  // We need to store local copies of the values pointed to by MArgs since they
  // may go out of scope before execution.
  std::vector<std::vector<char>> MArgStorage;

  // Stores auxiliary resources used by internal operations.
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;

  bool MIsEmpty = false;

  void register_successor(const std::shared_ptr<node_impl> &Node,
                          const std::shared_ptr<node_impl> &Prev) {
    MSuccessors.push_back(Node);
    Node->register_predecessor(Prev);
  }

  void register_predecessor(const std::shared_ptr<node_impl> &Node) {
    MPredecessors.push_back(Node);
  }

  node_impl() : MIsEmpty(true) {}

  node_impl(
      std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      sycl::detail::CG::CGTYPE CGType,
      const std::vector<sycl::detail::ArgDesc> &args,
      const std::vector<std::shared_ptr<const void>> &AuxiliaryResources)
      : MKernel(Kernel), MNDRDesc(NDRDesc), MOSModuleHandle(OSModuleHandle),
        MKernelName(KernelName), MAccStorage(AccStorage),
        MLocalAccStorage(LocalAccStorage), MRequirements(), MCGType(CGType),
        MArgs(args), MArgStorage(), MAuxiliaryResources(AuxiliaryResources) {

    // Need to copy the arg values to node local storage so that they don't go
    // out of scope before execution
    for (size_t i = 0; i < MArgs.size(); i++) {
      auto &CurrentArg = MArgs[i];
      MArgStorage.emplace_back(CurrentArg.MSize);
      auto StoragePtr = MArgStorage.back().data();
      if (CurrentArg.MPtr)
        std::memcpy(StoragePtr, CurrentArg.MPtr, CurrentArg.MSize);
      // Set the arg descriptor to point to the new storage
      CurrentArg.MPtr = StoragePtr;
      if (CurrentArg.MType ==
          sycl::detail::kernel_param_kind_t::kind_accessor) {
        MRequirements.push_back(
            static_cast<sycl::detail::AccessorImplHost *>(CurrentArg.MPtr));
      }
    }
  }

  // Recursively adding nodes to execution stack:
  void topology_sort(std::shared_ptr<node_impl> NodeImpl,
                     std::list<std::shared_ptr<node_impl>> &Schedule) {
    for (auto Next : MSuccessors) {
      // Check if we've already scheduled this node
      if (std::find(Schedule.begin(), Schedule.end(), Next) == Schedule.end())
        Next->topology_sort(Next, Schedule);
    }
    // We don't need to schedule empty nodes as they are only used when
    // calculating dependencies
    if (!NodeImpl->is_empty())
      Schedule.push_front(NodeImpl);
  }

  bool has_arg(const sycl::detail::ArgDesc &Arg) {
    for (auto &NodeArg : MArgs) {
      if (Arg.MType == NodeArg.MType && Arg.MSize == NodeArg.MSize) {
        // Args are actually void** so we need to dereference them to compare
        // actual values
        void *IncomingPtr = *static_cast<void **>(Arg.MPtr);
        void *ArgPtr = *static_cast<void **>(NodeArg.MPtr);
        if (IncomingPtr == ArgPtr) {
          return true;
        }
      }
    }
    return false;
  }

  bool is_empty() const { return MIsEmpty; }
};

struct graph_impl {

  graph_impl(const sycl::context &syclContext, const sycl::device &syclDevice)
      : MContext(syclContext), MDevice(syclDevice), MRecordingQueues(),
        MEventsMap() {}

  void add_root(const std::shared_ptr<node_impl> &);
  void remove_root(const std::shared_ptr<node_impl> &);

  std::shared_ptr<node_impl>
  add(std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      sycl::detail::CG::CGTYPE CGType,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<const void>> &AuxiliaryResources,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {},
      const std::vector<std::shared_ptr<sycl::detail::event_impl>> &DepEvents =
          {});

  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      std::function<void(handler &)> CGF,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  std::shared_ptr<node_impl>
  add(const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  void
  add_queue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.insert(RecordingQueue);
  }

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  void remove_queue(
      const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.erase(RecordingQueue);
  }

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state. \return True if any queues were
  /// removed.
  bool clear_queues();

  void add_event_for_node(std::shared_ptr<sycl::detail::event_impl> EventImpl,
                          std::shared_ptr<node_impl> NodeImpl) {
    MEventsMap[EventImpl] = NodeImpl;
  }

  std::shared_ptr<sycl::detail::event_impl>
  get_event_for_node(std::shared_ptr<node_impl> NodeImpl) const {
    if (auto EventImpl = std::find_if(
            MEventsMap.begin(), MEventsMap.end(),
            [NodeImpl](auto &it) { return it.second == NodeImpl; });
        EventImpl != MEventsMap.end()) {
      return EventImpl->first;
    }

    throw sycl::exception(
        sycl::errc::invalid,
        "No event has been recorded for the specified graph node");
  }

  /// Adds subgraph nodes from an executable graph to this graph, returns
  /// an empty node is used to schedule dependencies on this sub graph.
  std::shared_ptr<node_impl>
  add_subgraph_nodes(const std::list<std::shared_ptr<node_impl>> &NodeList);
  sycl::context get_context() const { return MContext; }

  std::set<std::shared_ptr<node_impl>> MRoots;
  std::shared_ptr<graph_impl> MParent;

private:
  // Context associated with this graph.
  sycl::context MContext;
  // Device associated with this graph. All graph nodes will execute on this
  // device.
  sycl::device MDevice;
  // Unique set of queues which are currently recording to this graph.
  std::set<std::shared_ptr<sycl::detail::queue_impl>> MRecordingQueues;
  // Map of events to their associated recorded nodes.
  std::unordered_map<std::shared_ptr<sycl::detail::event_impl>,
                     std::shared_ptr<node_impl>>
      MEventsMap;
};

class exec_graph_impl {
public:
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl)
      : MSchedule(), MGraphImpl(GraphImpl), MPiCommandBuffers(),
        MPiSyncPoints(), MContext(Context) {}
  ~exec_graph_impl();
  /// Add nodes to MSchedule
  void schedule();
  /// Enqueues the backend objects for the graph to the parametrized queue
  sycl::event enqueue(const std::shared_ptr<sycl::detail::queue_impl> &);
  /// Called by handler::ext_oneapi_command_graph() to schedule graph for
  /// execution
  sycl::event exec(const std::shared_ptr<sycl::detail::queue_impl> &);
  /// Turns our internal graph representation into PI command-buffers for a
  /// device
  void create_pi_command_buffers(sycl::device D);

  sycl::context get_context() const { return MContext; }

  const std::list<std::shared_ptr<node_impl>> &get_schedule() const {
    return MSchedule;
  }

private:
  RT::PiExtSyncPoint enqueue_node(sycl::context Ctx,
                                  sycl::detail::DeviceImplPtr DeviceImpl,
                                  RT::PiExtCommandBuffer CommandBuffer,
                                  std::shared_ptr<node_impl> Node);
  RT::PiExtSyncPoint enqueue_node_direct(sycl::context Ctx,
                                         sycl::detail::DeviceImplPtr DeviceImpl,
                                         RT::PiExtCommandBuffer CommandBuffer,
                                         std::shared_ptr<node_impl> Node);

  void find_real_deps(std::vector<RT::PiExtSyncPoint> &Deps,
                      std::shared_ptr<node_impl> CurrentNode);
  std::list<std::shared_ptr<node_impl>> MSchedule;
  // Pointer to the modifiable graph impl associated with this executable graph
  std::shared_ptr<graph_impl> MGraphImpl;
  // Map of devices to command buffers
  std::unordered_map<sycl::device, RT::PiExtCommandBuffer> MPiCommandBuffers;
  /// Map of nodes in the exec graph to the sync point representing their
  /// execution in the command graph.
  std::unordered_map<std::shared_ptr<node_impl>, RT::PiExtSyncPoint>
      MPiSyncPoints;
  // Context associated with this executable graph
  sycl::context MContext;
  // List of requirements for enqueueing this command graph, accumulated from
  // all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
