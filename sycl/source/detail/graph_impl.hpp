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
#include <deque>
#include <functional>
#include <list>
#include <set>
#include <shared_mutex>

namespace sycl {
inline namespace _V1 {

namespace detail {
class SYCLMemObjT;
}

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

  /// Used for tracking visited status during cycle checks.
  bool MVisited = false;

  /// Add successor to the node.
  /// @param Node Node to add as a successor.
  /// @param Prev Predecessor to \p node being added as successor.
  ///
  /// \p Prev should be a shared_ptr to an instance of this object, but can't
  /// use a raw \p this pointer, so the extra \Prev parameter is passed.
  void registerSuccessor(const std::shared_ptr<node_impl> &Node,
                         const std::shared_ptr<node_impl> &Prev) {
    if (std::find(MSuccessors.begin(), MSuccessors.end(), Node) !=
        MSuccessors.end()) {
      return;
    }
    MSuccessors.push_back(Node);
    Node->registerPredecessor(Prev);
  }

  /// Add predecessor to the node.
  /// @param Node Node to add as a predecessor.
  void registerPredecessor(const std::shared_ptr<node_impl> &Node) {
    if (std::find_if(MPredecessors.begin(), MPredecessors.end(),
                     [&Node](const std::weak_ptr<node_impl> &Ptr) {
                       return Ptr.lock() == Node;
                     }) != MPredecessors.end()) {
      return;
    }
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

  /// Tests if two nodes have the same content,
  /// i.e. same command group
  /// This function should only be used for internal purposes.
  /// A true return from this operator is not a guarantee that the nodes are
  /// equals according to the Common reference semantics. But this function is
  /// an helper to verify that two nodes contain equivalent Command Groups.
  /// @param Node node to compare with
  /// @return true if two nodes have equivament command groups. false otherwise.
  bool operator==(const node_impl &Node) {
    if (MCGType != Node.MCGType)
      return false;

    switch (MCGType) {
    case sycl::detail::CG::CGTYPE::Kernel: {
      sycl::detail::CGExecKernel *ExecKernelA =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      sycl::detail::CGExecKernel *ExecKernelB =
          static_cast<sycl::detail::CGExecKernel *>(Node.MCommandGroup.get());
      return ExecKernelA->MKernelName.compare(ExecKernelB->MKernelName) == 0;
    }
    case sycl::detail::CG::CGTYPE::CopyUSM: {
      sycl::detail::CGCopyUSM *CopyA =
          static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
      sycl::detail::CGCopyUSM *CopyB =
          static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
      return (CopyA->getSrc() == CopyB->getSrc()) &&
             (CopyA->getDst() == CopyB->getDst()) &&
             (CopyA->getLength() == CopyB->getLength());
    }
    case sycl::detail::CG::CGTYPE::CopyAccToAcc:
    case sycl::detail::CG::CGTYPE::CopyAccToPtr:
    case sycl::detail::CG::CGTYPE::CopyPtrToAcc: {
      sycl::detail::CGCopy *CopyA =
          static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
      sycl::detail::CGCopy *CopyB =
          static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
      return (CopyA->getSrc() == CopyB->getSrc()) &&
             (CopyA->getDst() == CopyB->getDst());
    }
    default:
      assert(false && "Unexpected command group type!");
      return false;
    }
  }

  /// Recursively add nodes to execution stack.
  /// @param NodeImpl Node to schedule.
  /// @param Schedule Execution ordering to add node to.
  void sortTopological(std::shared_ptr<node_impl> NodeImpl,
                       std::list<std::shared_ptr<node_impl>> &Schedule) {
    for (auto &Next : MSuccessors) {
      // Check if we've already scheduled this node
      if (std::find(Schedule.begin(), Schedule.end(), Next) == Schedule.end())
        Next->sortTopological(Next, Schedule);
    }

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
    case sycl::detail::CG::CopyImage:
      return createCGCopy<sycl::detail::CGCopyImage>();
    case sycl::detail::CG::SemaphoreSignal:
      return createCGCopy<sycl::detail::CGSemaphoreSignal>();
    case sycl::detail::CG::SemaphoreWait:
      return createCGCopy<sycl::detail::CGSemaphoreWait>();
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

  /// Tests is the caller is similar to Node
  /// @return True if the two nodes are similar
  bool isSimilar(std::shared_ptr<node_impl> Node) {
    if (MSuccessors.size() != Node->MSuccessors.size())
      return false;

    if (MPredecessors.size() != Node->MPredecessors.size())
      return false;

    if (*this == *Node.get())
      return true;

    return false;
  }

  /// Recursive traversal of successor nodes checking for
  /// equivalent node successions in Node
  /// @param Node pointer to the starting node for structure comparison
  /// @return true is same structure found, false otherwise
  bool checkNodeRecursive(std::shared_ptr<node_impl> Node) {
    size_t FoundCnt = 0;
    for (std::shared_ptr<node_impl> SuccA : MSuccessors) {
      for (std::shared_ptr<node_impl> SuccB : Node->MSuccessors) {
        if (isSimilar(Node) && SuccA->checkNodeRecursive(SuccB)) {
          FoundCnt++;
          break;
        }
      }
    }
    if (FoundCnt != MSuccessors.size()) {
      return false;
    }

    return true;
  }

  /// Recusively computes the number of successor nodes
  /// @return number of successor nodes
  size_t depthSearchCount() const {
    size_t NumberOfNodes = 1;
    for (const auto &Succ : MSuccessors) {
      NumberOfNodes += Succ->depthSearchCount();
    }
    return NumberOfNodes;
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
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device to create nodes with.
  /// @param PropList Optional list of properties.
  graph_impl(const sycl::context &SyclContext, const sycl::device &SyclDevice,
             const sycl::property_list &PropList = {})
      : MContext(SyclContext), MDevice(SyclDevice), MRecordingQueues(),
        MEventsMap(), MInorderQueueMap() {
    if (PropList.has_property<property::graph::no_cycle_check>()) {
      MSkipCycleChecks = true;
    }
    if (PropList
            .has_property<property::graph::assume_buffer_outlives_graph>()) {
      MAllowBuffers = true;
    }

    if (SyclDevice.get_info<
            ext::oneapi::experimental::info::device::graph_support>() ==
        graph_support_level::unsupported) {
      std::stringstream Stream;
      Stream << SyclDevice.get_backend();
      std::string BackendString = Stream.str();
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          BackendString + " backend is not supported by SYCL Graph extension.");
    }
  }

  ~graph_impl();

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
    ReadLock Lock(MMutex);
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

  /// Duplicates and Adds sub-graph nodes from an executable graph to this
  /// graph.
  /// @param SubGraphExec sub-graph to add to the parent.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addSubgraphNodes(const std::shared_ptr<exec_graph_impl> &SubGraphExec);

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

  /// Make an edge between two nodes in the graph. Performs some mandatory
  /// error checks as well as an optional check for cycles introduced by making
  /// this edge.
  /// @param Src The source of the new edge.
  /// @param Dest The destination of the new edge.
  void makeEdge(std::shared_ptr<node_impl> Src,
                std::shared_ptr<node_impl> Dest);

  /// Throws an invalid exception if this function is called
  /// while a queue is recording commands to the graph.
  /// @param ExceptionMsg Message to append to the exception message
  void throwIfGraphRecordingQueue(const std::string ExceptionMsg) const {
    if (MRecordingQueues.size()) {
      throw sycl::exception(make_error_code(sycl::errc::invalid),
                            ExceptionMsg +
                                " cannot be called when a queue "
                                "is currently recording commands to a graph.");
    }
  }

  /// Checks if the graph_impl of Graph has a similar structure to
  /// the graph_impl of the caller.
  /// Graphs are considered similar if they have same numbers of nodes
  /// of the same type with similar predecessor and successor nodes (number and
  /// type). Two nodes are considered similar if they have the same
  /// command-group type. For command-groups of type "kernel", the "signature"
  /// of the kernel is also compared (i.e. the name of the command-group).
  /// @param Graph if reference to the graph to compare with.
  /// @param DebugPrint if set to true throw exception with additional debug
  /// information about the spotted graph differences.
  /// @return true if the two graphs are similar, false otherwise
  bool hasSimilarStructure(std::shared_ptr<detail::graph_impl> Graph,
                           bool DebugPrint = false) const {
    if (this == Graph.get())
      return true;

    if (MContext != Graph->MContext) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MContext are not the same.");
      }
      return false;
    }

    if (MDevice != Graph->MDevice) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MDevice are not the same.");
      }
      return false;
    }

    if (MEventsMap.size() != Graph->MEventsMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MEventsMap sizes are not the same.");
      }
      return false;
    }

    if (MInorderQueueMap.size() != Graph->MInorderQueueMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MInorderQueueMap sizes are not the same.");
      }
      return false;
    }

    if (MRoots.size() != Graph->MRoots.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MRoots sizes are not the same.");
      }
      return false;
    }

    size_t RootsFound = 0;
    for (std::shared_ptr<node_impl> NodeA : MRoots) {
      for (std::shared_ptr<node_impl> NodeB : Graph->MRoots) {
        if (NodeA->isSimilar(NodeB)) {
          if (NodeA->checkNodeRecursive(NodeB)) {
            RootsFound++;
            break;
          }
        }
      }
    }

    if (RootsFound != MRoots.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "Root Nodes do NOT match.");
      }
      return false;
    }

    return true;
  }

  // Returns the number of nodes in the Graph
  // @return Number of nodes in the Graph
  size_t getNumberOfNodes() const {
    size_t NumberOfNodes = 0;
    for (const auto &Node : MRoots) {
      NumberOfNodes += Node->depthSearchCount();
    }
    return NumberOfNodes;
  }

private:
  /// Iterate over the graph depth-first and run \p NodeFunc on each node.
  /// @param NodeFunc A function which receives as input a node in the graph to
  /// perform operations on as well as the stack of nodes encountered in the
  /// current path. The return value of this function determines whether an
  /// early exit is triggered, if true the depth-first search will end
  /// immediately and no further nodes will be visited.
  void
  searchDepthFirst(std::function<bool(std::shared_ptr<node_impl> &,
                                      std::deque<std::shared_ptr<node_impl>> &)>
                       NodeFunc);

  /// Check the graph for cycles by performing a depth-first search of the
  /// graph. If a node is visited more than once in a given path through the
  /// graph, a cycle is present and the search ends immediately.
  /// @return True if a cycle is detected, false if not.
  bool checkForCycles();

  /// Insert node into list of root nodes.
  /// @param Root Node to add to list of root nodes.
  void addRoot(const std::shared_ptr<node_impl> &Root);

  /// Adds nodes to the exit nodes of this graph.
  /// @param NodeList List of nodes from sub-graph in schedule order.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addNodesToExits(const std::list<std::shared_ptr<node_impl>> &NodeList);

  /// Context associated with this graph.
  sycl::context MContext;
  /// Device associated with this graph. All graph nodes will execute on this
  /// device.
  sycl::device MDevice;
  /// Unique set of queues which are currently recording to this graph.
  std::set<std::weak_ptr<sycl::detail::queue_impl>,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MRecordingQueues;
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
  /// Controls whether we skip the cycle checks in makeEdge, set by the presence
  /// of the no_cycle_check property on construction.
  bool MSkipCycleChecks = false;
  /// Unique set of SYCL Memory Objects which are currently in use in the graph.
  std::set<sycl::detail::SYCLMemObjT *> MMemObjs;

  /// Controls whether we allow buffers to be used in the graph. Set by the
  /// presence of the assume_buffer_outlives_graph property.
  bool MAllowBuffers = false;
};

/// Class representing the implementation of command_graph<executable>.
class exec_graph_impl {
public:
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

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

  /// Query the graph_impl.
  /// @return pointer to the graph_impl MGraphImpl
  const std::shared_ptr<graph_impl> &getGraphImpl() const { return MGraphImpl; }

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
  /// Thread-safe implementation note: in the current implementation
  /// multiple exec_graph_impl can reference the same graph_impl object.
  /// This specificity must be taken into account when trying to lock
  /// the graph_impl mutex from an exec_graph_impl to avoid deadlock.
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
  /// Storage for accessors which are used by this graph, accumulated from
  /// all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplPtr> MAccessors;
  /// List of all execution events returned from command buffer enqueue calls.
  std::vector<sycl::detail::EventImplPtr> MExecutionEvents;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
