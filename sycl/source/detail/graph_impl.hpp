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

#include <detail/accessor_impl.hpp>
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

/// Implementation of node class from SYCL_EXT_ONEAPI_GRAPH.
class node_impl {
public:
  /// List of successors to this node.
  std::vector<std::shared_ptr<node_impl>> MSuccessors;
  /// List of predecessors to this node.
  ///
  /// Using weak_ptr here to prevent circular references between nodes.
  std::vector<std::weak_ptr<node_impl>> MPredecessors;
  /// Type of the command-group for the node.
  sycl::detail::CG::CGTYPE MCGType = sycl::detail::CG::None;
  /// Command group object which stores all args etc needed to enqueue the node
  std::unique_ptr<sycl::detail::CG> MCommandGroup;

  /// Add successor to the node.
  /// @param Node Node to add as a successor.
  /// @param Prev Predecessor to \p node being added as successor.
  ///
  /// \p Prev should be a shared_ptr to an instance of this object, but can't
  /// use a raw \p this pointer, so the extra \Prev parameter is passed.
  void registerSuccessor(const std::shared_ptr<node_impl> &Node,
                         const std::shared_ptr<node_impl> &Prev) {
    MSuccessors.push_back(Node);
    Node->registerPredecessor(Prev);
  }

  /// Add predecessor to the node.
  /// @param Node Node to add as a predecessor.
  void registerPredecessor(const std::shared_ptr<node_impl> &Node) {
    MPredecessors.push_back(Node);
  }

  /// Construct an empty node.
  node_impl() {}

  /// Construct a node representing a command-group.
  /// @param CGType Type of the command-group.
  /// @param CommandGroup The CG which stores the command information for this
  /// node.
  node_impl(sycl::detail::CG::CGTYPE CGType,
            std::unique_ptr<sycl::detail::CG> &&CommandGroup)
      : MCGType(CGType), MCommandGroup(std::move(CommandGroup)) {}

  /// Recursively add nodes to execution stack.
  /// @param NodeImpl Node to schedule.
  /// @param Schedule Execution ordering to add node to.
  void sortTopological(std::shared_ptr<node_impl> NodeImpl,
                       std::list<std::shared_ptr<node_impl>> &Schedule) {
    for (auto Next : MSuccessors) {
      // Check if we've already scheduled this node
      if (std::find(Schedule.begin(), Schedule.end(), Next) == Schedule.end())
        Next->sortTopological(Next, Schedule);
    }
    // We don't need to schedule empty nodes as they are only used when
    // calculating dependencies
    if (!NodeImpl->isEmpty())
      Schedule.push_front(NodeImpl);
  }

  /// Checks if this node has a given requirement.
  /// @param Requirement Requirement to lookup.
  /// @return True if \p Requirement is present in node, false otherwise.
  bool hasRequirement(sycl::detail::AccessorImplHost *IncomingReq) {
    for (sycl::detail::AccessorImplHost *CurrentReq :
         MCommandGroup->getRequirements()) {
      if (IncomingReq->MSYCLMemObj == CurrentReq->MSYCLMemObj) {
        return true;
      }
    }
    return false;
  }

  /// Query if this is an empty node.
  /// @return True if this is an empty node, false otherwise.
  bool isEmpty() const { return MCGType == sycl::detail::CG::None; }

  /// Get a deep copy of this node's command group
  /// @return A unique ptr to the new command group object.
  std::unique_ptr<sycl::detail::CG> getCGCopy() const {
    switch (MCGType) {
    case sycl::detail::CG::Kernel:
      return createCGCopy<sycl::detail::CGExecKernel>();
    case sycl::detail::CG::CopyAccToPtr:
    case sycl::detail::CG::CopyPtrToAcc:
    case sycl::detail::CG::CopyAccToAcc:
      return createCGCopy<sycl::detail::CGCopy>();
    case sycl::detail::CG::Fill:
      return createCGCopy<sycl::detail::CGFill>();
    case sycl::detail::CG::UpdateHost:
      return createCGCopy<sycl::detail::CGUpdateHost>();
    case sycl::detail::CG::CopyUSM:
      return createCGCopy<sycl::detail::CGCopyUSM>();
    case sycl::detail::CG::FillUSM:
      return createCGCopy<sycl::detail::CGFillUSM>();
    case sycl::detail::CG::PrefetchUSM:
      return createCGCopy<sycl::detail::CGPrefetchUSM>();
    case sycl::detail::CG::AdviseUSM:
      return createCGCopy<sycl::detail::CGAdviseUSM>();
    case sycl::detail::CG::Copy2DUSM:
      return createCGCopy<sycl::detail::CGCopy2DUSM>();
    case sycl::detail::CG::Fill2DUSM:
      return createCGCopy<sycl::detail::CGFill2DUSM>();
    case sycl::detail::CG::Memset2DUSM:
      return createCGCopy<sycl::detail::CGMemset2DUSM>();
    case sycl::detail::CG::CodeplayHostTask:
      assert(false);
      break;
      // TODO: Uncomment this once we implement support for host task so we can
      // test required changes to the CG class.

      // return createCGCopy<sycl::detail::CGHostTask>();
    case sycl::detail::CG::Barrier:
    case sycl::detail::CG::BarrierWaitlist:
      return createCGCopy<sycl::detail::CGBarrier>();
    case sycl::detail::CG::CopyToDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyToDeviceGlobal>();
    case sycl::detail::CG::CopyFromDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyFromDeviceGlobal>();
    case sycl::detail::CG::ReadWriteHostPipe:
      return createCGCopy<sycl::detail::CGReadWriteHostPipe>();
    case sycl::detail::CG::ExecCommandBuffer:
      assert(false &&
             "Error: Command graph submission should not be a node in a graph");
      break;
    case sycl::detail::CG::None:
      assert(false &&
             "Error: Empty nodes should not be enqueue to a command buffer");
      break;
    }
    return nullptr;
  }

private:
  /// Creates a copy of the node's CG by casting to it's actual type, then using
  /// that to copy construct and create a new unique ptr from that copy.
  /// @tparam CGT The derived type of the CG.
  /// @return A new unique ptr to the copied CG.
  template <typename CGT> std::unique_ptr<CGT> createCGCopy() const {
    return std::make_unique<CGT>(*static_cast<CGT *>(MCommandGroup.get()));
  }
};

/// Implementation details of command_graph<modifiable>.
class graph_impl {
public:
  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device to create nodes with.
  graph_impl(const sycl::context &SyclContext, const sycl::device &SyclDevice)
      : MContext(SyclContext), MDevice(SyclDevice), MRecordingQueues(),
        MEventsMap(), MInorderQueueMap() {}

  /// Insert node into list of root nodes.
  /// @param Root Node to add to list of root nodes.
  void addRoot(const std::shared_ptr<node_impl> &Root);

  /// Remove node from list of root nodes.
  /// @param Root Node to remove from list of root nodes.
  void removeRoot(const std::shared_ptr<node_impl> &Root);

  /// Create a kernel node in the graph.
  /// @param CGType Type of the command-group.
  /// @param CommandGroup The CG which stores all information for this node.
  /// @param Dep Dependencies of the created node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(sycl::detail::CG::CGTYPE CGType,
      std::unique_ptr<sycl::detail::CG> CommandGroup,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create a CGF node in the graph.
  /// @param Impl Graph implementation pointer to create a handler with.
  /// @param CGF Command-group function to create node with.
  /// @param Args Node arguments.
  /// @param Dep Dependencies of the created node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      std::function<void(handler &)> CGF,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create an empty node in the graph.
  /// @param Dep List of predecessor nodes.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create an empty node in the graph.
  /// @param Events List of events associated to this node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::vector<sycl::detail::EventImplPtr> Events);

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  /// @param RecordingQueue Queue to add to set.
  void
  addQueue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.insert(RecordingQueue);
  }

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  /// @param RecordingQueue Queue to remove from set.
  void
  removeQueue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.erase(RecordingQueue);
  }

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state.
  ///
  /// @return True if any queues were removed.
  bool clearQueues();

  /// Associate a sycl event with a node in the graph.
  /// @param EventImpl Event to associate with a node in map.
  /// @param NodeImpl Node to associate with event in map.
  void addEventForNode(std::shared_ptr<sycl::detail::event_impl> EventImpl,
                       std::shared_ptr<node_impl> NodeImpl) {
    MEventsMap[EventImpl] = NodeImpl;
  }

  /// Find the sycl event associated with a node.
  /// @param NodeImpl Node to find event for.
  /// @return Event associated with node.
  std::shared_ptr<sycl::detail::event_impl>
  getEventForNode(std::shared_ptr<node_impl> NodeImpl) const {
    if (auto EventImpl = std::find_if(
            MEventsMap.begin(), MEventsMap.end(),
            [NodeImpl](auto &it) { return it.second == NodeImpl; });
        EventImpl != MEventsMap.end()) {
      return EventImpl->first;
    }

    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "No event has been recorded for the specified graph node");
  }

  /// Adds sub-graph nodes from an executable graph to this graph.
  /// @param NodeList List of nodes from sub-graph in schedule order.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addSubgraphNodes(const std::list<std::shared_ptr<node_impl>> &NodeList);

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

  /// Query for the device tied to this graph.
  /// @return Device associated with graph.
  sycl::device getDevice() const { return MDevice; }

  /// List of root nodes.
  std::set<std::shared_ptr<node_impl>> MRoots;

  /// Find the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to find the last node added to the graph from.
  /// @return Last node in this graph added from \p Queue recording, or empty
  /// shared pointer if none.
  std::shared_ptr<node_impl>
  getLastInorderNode(std::shared_ptr<sycl::detail::queue_impl> Queue) {
    std::weak_ptr<sycl::detail::queue_impl> QueueWeakPtr(Queue);
    if (0 == MInorderQueueMap.count(QueueWeakPtr)) {
      return {};
    }
    return MInorderQueueMap[QueueWeakPtr];
  }

  /// Track the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to register \p Node for.
  /// @param Node Last node that was added to this graph from \p Queue.
  void setLastInorderNode(std::shared_ptr<sycl::detail::queue_impl> Queue,
                          std::shared_ptr<node_impl> Node) {
    std::weak_ptr<sycl::detail::queue_impl> QueueWeakPtr(Queue);
    MInorderQueueMap[QueueWeakPtr] = Node;
  }

private:
  /// Context associated with this graph.
  sycl::context MContext;
  /// Device associated with this graph. All graph nodes will execute on this
  /// device.
  sycl::device MDevice;
  /// Unique set of queues which are currently recording to this graph.
  std::set<std::shared_ptr<sycl::detail::queue_impl>> MRecordingQueues;
  /// Map of events to their associated recorded nodes.
  std::unordered_map<std::shared_ptr<sycl::detail::event_impl>,
                     std::shared_ptr<node_impl>>
      MEventsMap;
  /// Map for every in-order queue thats recorded a node to the graph, what
  /// the last node added was. We can use this to create new edges on the last
  /// node if any more nodes are added to the graph from the queue.
  std::map<std::weak_ptr<sycl::detail::queue_impl>, std::shared_ptr<node_impl>,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MInorderQueueMap;
};

/// Class representing the implementation of command_graph<executable>.
class exec_graph_impl {
public:
  /// Constructor.
  /// @param Context Context to create graph with.
  /// @param GraphImpl Modifiable graph implementation to create with.
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl)
      : MSchedule(), MGraphImpl(GraphImpl), MPiCommandBuffers(),
        MPiSyncPoints(), MContext(Context), MRequirements(),
        MExecutionEvents() {}

  /// Destructor.
  ///
  /// Releases any PI command-buffers the object has created.
  ~exec_graph_impl();

  /// Add nodes to MSchedule.
  void schedule();

  /// Called by handler::ext_oneapi_command_graph() to schedule graph for
  /// execution.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data provided by the sycl::handler
  /// @return Event associated with the execution of the graph.
  sycl::event enqueue(const std::shared_ptr<sycl::detail::queue_impl> &Queue,
                      sycl::detail::CG::StorageInitHelper CGData);

  /// Turns the internal graph representation into UR command-buffers for a
  /// device.
  /// @param Device Device to create backend command-buffers for.
  void createCommandBuffers(sycl::device Device);

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

  /// Query the scheduling of node execution.
  /// @return List of nodes in execution order.
  const std::list<std::shared_ptr<node_impl>> &getSchedule() const {
    return MSchedule;
  }

private:
  /// Create a command-group for the node and add it to command-buffer by going
  /// through the scheduler.
  /// @param Ctx Context to use.
  /// @param DeviceImpl Device associated with the enqueue.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @return PI sync point created for this node in the command-buffer.
  sycl::detail::pi::PiExtSyncPoint
  enqueueNode(sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
              sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
              std::shared_ptr<node_impl> Node);

  /// Enqueue a node directly to the command-buffer without going through the
  /// scheduler.
  /// @param Ctx Context to use.
  /// @param DeviceImpl Device associated with the enqueue.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @return PI sync point created for this node in the command-buffer.
  sycl::detail::pi::PiExtSyncPoint
  enqueueNodeDirect(sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
                    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
                    std::shared_ptr<node_impl> Node);

  /// Iterates back through predecessors to find the real dependency.
  /// @param[out] Deps Found dependencies.
  /// @param[in] CurrentNode Node to find dependencies for.
  void findRealDeps(std::vector<sycl::detail::pi::PiExtSyncPoint> &Deps,
                    std::shared_ptr<node_impl> CurrentNode);

  /// Execution schedule of nodes in the graph.
  std::list<std::shared_ptr<node_impl>> MSchedule;
  /// Pointer to the modifiable graph impl associated with this executable
  /// graph.
  std::shared_ptr<graph_impl> MGraphImpl;
  /// Map of devices to command buffers.
  std::unordered_map<sycl::device, sycl::detail::pi::PiExtCommandBuffer>
      MPiCommandBuffers;
  /// Map of nodes in the exec graph to the sync point representing their
  /// execution in the command graph.
  std::unordered_map<std::shared_ptr<node_impl>,
                     sycl::detail::pi::PiExtSyncPoint>
      MPiSyncPoints;
  /// Context associated with this executable graph.
  sycl::context MContext;
  /// List of requirements for enqueueing this command graph, accumulated from
  /// all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;
  /// List of all execution events returned from command buffer enqueue calls.
  std::vector<sycl::detail::EventImplPtr> MExecutionEvents;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
