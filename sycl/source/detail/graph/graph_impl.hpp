//==--------- graph_impl.hpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "memory_pool.hpp"          // for graph_mem_pool
#include "node_impl.hpp"            // for node_impl
#include <detail/event_impl.hpp>    // for event_impl
#include <sycl/detail/cg_types.hpp> // for CGType
#include <sycl/detail/os_util.hpp>  // for OS utils

#include <fstream>      // for fstream
#include <functional>   // for function
#include <list>         // for list
#include <memory>       // for shared_ptr
#include <optional>     // for optional
#include <set>          // for set
#include <shared_mutex> // for shared_mutex
#include <vector>       // for vector

// For testing of graph internals
class GraphImplTest;

namespace sycl {
inline namespace _V1 {
// Forward declarations
class handler;

// Forward declarations
namespace detail {
class SYCLMemObjT;
class queue_impl;
class NDRDescT;
class ArgDesc;
class CG;
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
// Forward declarations
enum class graph_state;
template <graph_state State> class command_graph;

namespace detail {
// Forward declarations
class dynamic_command_group_impl;

class partition {
public:
  /// Constructor.
  partition() : MSchedule(), MCommandBuffers() {}

  /// List of root nodes.
  std::set<node_impl *> MRoots;
  /// Execution schedule of nodes in the graph.
  std::list<node_impl *> MSchedule;
  /// Map of devices to command buffers.
  std::unordered_map<sycl::device, ur_exp_command_buffer_handle_t>
      MCommandBuffers;
  /// List of predecessors to this partition.
  std::vector<partition *> MPredecessors;

  /// List of successors to this partition.
  std::vector<partition *> MSuccessors;

  /// List of requirements for this partition.
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;

  /// Storage for accessors which are used by this partition.
  std::vector<AccessorImplPtr> MAccessors;

  /// True if the graph of this partition is a single path graph
  /// and in-order optmization can be applied on it.
  bool MIsInOrderGraph = false;

  /// True if this partition contains only one node which is a host_task.
  bool MIsHostTask = false;

  // Submission event for the partition. Used during enqueue to define
  // dependencies between this partition and its successors. This event is
  // replaced every time the partition is executed.
  EventImplPtr MEvent;

  nodes_range roots() const { return MRoots; }
  nodes_range schedule() const { return MSchedule; }

  /// Checks if the graph is single path, i.e. each node has a single successor.
  /// @return True if the graph is a single path
  bool checkIfGraphIsSinglePath() {
    if (MRoots.size() > 1) {
      return false;
    }
    for (node_impl &Node : schedule()) {
      // In version 1.3.28454 of the L0 driver, 2D Copy ops cannot not
      // be enqueued in an in-order cmd-list (causing execution to stall).
      // The 2D Copy test should be removed from here when the bug is fixed.
      if ((Node.MSuccessors.size() > 1) || (Node.isNDCopyNode())) {
        return false;
      }
    }

    return true;
  }

  /// Add nodes to MSchedule.
  void updateSchedule();
};

/// Implementation details of command_graph<modifiable>.
class graph_impl : public std::enable_shared_from_this<graph_impl> {
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
             const sycl::property_list &PropList = {});

  ~graph_impl();

  /// Remove node from list of root nodes.
  /// @param Root Node to remove from list of root nodes.
  void removeRoot(node_impl &Root);

  /// Verifies the CG is valid to add to the graph and returns set of
  /// dependent nodes if so.
  /// @param CommandGroup The command group to verify and retrieve edges for.
  /// @return Set of dependent nodes in the graph.
  std::set<node_impl *>
  getCGEdges(const std::shared_ptr<sycl::detail::CG> &CommandGroup) const;

  /// Identifies the sycl buffers used in the command-group and marks them
  /// as used in the graph.
  /// @param CommandGroup The command-group to check for buffer usage in.
  void markCGMemObjs(const std::shared_ptr<sycl::detail::CG> &CommandGroup);

  /// Create a kernel node in the graph.
  /// @param NodeType User facing type of the node.
  /// @param CommandGroup The CG which stores all information for this node.
  /// @param Deps Dependencies of the created node.
  /// @return Created node in the graph.
  node_impl &add(node_type NodeType,
                 std::shared_ptr<sycl::detail::CG> CommandGroup,
                 nodes_range Deps);

  /// Create a CGF node in the graph.
  /// @param CGF Command-group function to create node with.
  /// @param Args Node arguments.
  /// @param Deps Dependencies of the created node.
  /// @return Created node in the graph.
  node_impl &add(std::function<void(handler &)> CGF,
                 const std::vector<sycl::detail::ArgDesc> &Args,
                 nodes_range Deps);

  /// Create an empty node in the graph.
  /// @param Deps List of predecessor nodes.
  /// @return Created node in the graph.
  node_impl &add(nodes_range Deps);

  /// Create a dynamic command-group node in the graph.
  /// @param DynCGImpl Dynamic command-group used to create node.
  /// @param Deps List of predecessor nodes.
  /// @return Created node in the graph.
  node_impl &add(std::shared_ptr<dynamic_command_group_impl> &DynCGImpl,
                 nodes_range Deps);

  std::shared_ptr<sycl::detail::queue_impl> getQueue() const;

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  /// @param RecordingQueue Queue to add to set.
  void addQueue(sycl::detail::queue_impl &RecordingQueue);

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  /// @param RecordingQueue Queue to remove from set.
  void removeQueue(sycl::detail::queue_impl &RecordingQueue);

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state.
  void clearQueues(bool NeedsLock);

  /// Associate a sycl event with a node in the graph.
  /// @param EventImpl Event to associate with a node in map.
  /// @param NodeImpl Node to associate with event in map.
  void addEventForNode(std::shared_ptr<sycl::detail::event_impl> EventImpl,
                       node_impl &NodeImpl) {
    if (!(EventImpl->hasCommandGraph()))
      EventImpl->setCommandGraph(shared_from_this());
    MEventsMap[EventImpl] = &NodeImpl;
  }

  /// Find the sycl event associated with a node.
  /// @param NodeImpl Node to find event for.
  /// @return Event associated with node.
  std::shared_ptr<sycl::detail::event_impl>
  getEventForNode(node_impl &NodeImpl) const {
    ReadLock Lock(MMutex);
    if (auto EventImpl = std::find_if(
            MEventsMap.begin(), MEventsMap.end(),
            [&NodeImpl](auto &it) { return it.second == &NodeImpl; });
        EventImpl != MEventsMap.end()) {
      return EventImpl->first;
    }

    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "No event has been recorded for the specified graph node");
  }

  /// Find the node associated with a SYCL event. Throws if no node is found for
  /// the given event.
  /// @param EventImpl Event to find the node for.
  /// @return Node associated with the event.
  node_impl &
  getNodeForEvent(std::shared_ptr<sycl::detail::event_impl> EventImpl) {
    ReadLock Lock(MMutex);

    if (auto NodeFound = MEventsMap.find(EventImpl);
        NodeFound != std::end(MEventsMap)) {
      // TODO: Is it guaranteed to be non-null?
      return *NodeFound->second;
    }

    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "No node in this graph is associated with this event");
  }

  /// Find the nodes associated with a list of SYCL events. Throws if no node is
  /// found for a given event.
  /// @param Events Events to find nodes for.
  /// @return A list of node counterparts for each event, in the same order.
  std::vector<node_impl *> getNodesForEvents(
      const std::vector<std::shared_ptr<sycl::detail::event_impl>> &Events) {
    std::vector<node_impl *> NodeList{};
    NodeList.reserve(Events.size());

    ReadLock Lock(MMutex);

    for (const auto &Event : Events) {
      if (auto NodeFound = MEventsMap.find(Event);
          NodeFound != std::end(MEventsMap)) {
        NodeList.push_back(NodeFound->second);
      } else {
        throw sycl::exception(
            sycl::make_error_code(errc::invalid),
            "No node in this graph is associated with this event");
      }
    }

    return NodeList;
  }

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Query for the context impl tied to this graph.
  /// @return shared_ptr ref for the context impl associated with graph.
  const std::shared_ptr<sycl::detail::context_impl> &getContextImplPtr() const {
    return sycl::detail::getSyclObjImpl(MContext);
  }
#endif
  sycl::detail::context_impl &getContextImpl() const {
    return *sycl::detail::getSyclObjImpl(MContext);
  }

  /// Query for the device_impl tied to this graph.
  /// @return device_impl shared ptr reference associated with graph.
  device_impl &getDeviceImpl() const { return *getSyclObjImpl(MDevice); }

  /// Query for the device tied to this graph.
  /// @return Device associated with graph.
  sycl::device getDevice() const { return MDevice; }

  /// List of root nodes.
  std::set<node_impl *> MRoots;

  /// Storage for all nodes contained within a graph. Nodes are connected to
  /// each other via weak_ptrs and so do not extend each other's lifetimes.
  /// This storage allows easy iteration over all nodes in the graph, rather
  /// than needing an expensive depth first search.
  std::vector<std::shared_ptr<node_impl>> MNodeStorage;

  nodes_range roots() const { return MRoots; }
  nodes_range nodes() const { return MNodeStorage; }

  /// Find the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to find the last node added to the graph from.
  /// @return Last node in this graph added from \p Queue recording, or empty
  /// shared pointer if none.
  node_impl *getLastInorderNode(sycl::detail::queue_impl *Queue);

  /// Track the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to register \p Node for.
  /// @param Node Last node that was added to this graph from \p Queue.
  void setLastInorderNode(sycl::detail::queue_impl &Queue, node_impl &Node);

  /// Prints the contents of the graph to a text file in DOT format.
  /// @param FilePath Path to the output file.
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printGraphAsDot(const std::string FilePath, bool Verbose) const {
    /// Vector of nodes visited during the graph printing
    std::vector<node_impl *> VisitedNodes;

    std::fstream Stream(FilePath, std::ios::out);
    Stream << "digraph dot {" << std::endl;

    for (node_impl &Node : roots())
      Node.printDotRecursive(Stream, VisitedNodes, Verbose);

    Stream << "}" << std::endl;

    Stream.close();
  }

  /// Make an edge between two nodes in the graph. Performs some mandatory
  /// error checks as well as an optional check for cycles introduced by making
  /// this edge.
  /// @param Src The source of the new edge.
  /// @param Dest The destination of the new edge.
  void makeEdge(node_impl &Src, node_impl &Dest);

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

  /// Recursively check successors of NodeA and NodeB to check they are similar.
  /// @param NodeA pointer to the first node for comparison
  /// @param NodeB pointer to the second node for comparison
  /// @return true is same structure found, false otherwise
  static bool checkNodeRecursive(node_impl &NodeA, node_impl &NodeB) {
    size_t FoundCnt = 0;
    for (node_impl &SuccA : NodeA.successors()) {
      for (node_impl &SuccB : NodeB.successors()) {
        if (NodeA.isSimilar(NodeB) && checkNodeRecursive(SuccA, SuccB)) {
          FoundCnt++;
          break;
        }
      }
    }
    if (FoundCnt != NodeA.MSuccessors.size()) {
      return false;
    }

    return true;
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
  bool hasSimilarStructure(detail::graph_impl &Graph,
                           bool DebugPrint = false) const {
    if (this == &Graph)
      return true;

    if (MContext != Graph.MContext) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MContext are not the same.");
      }
      return false;
    }

    if (MDevice != Graph.MDevice) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MDevice are not the same.");
      }
      return false;
    }

    if (MEventsMap.size() != Graph.MEventsMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MEventsMap sizes are not the same.");
      }
      return false;
    }

    if (MInorderQueueMap.size() != Graph.MInorderQueueMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MInorderQueueMap sizes are not the same.");
      }
      return false;
    }

    if (MRoots.size() != Graph.MRoots.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MRoots sizes are not the same.");
      }
      return false;
    }

    size_t RootsFound = 0;
    for (node_impl &NodeA : roots()) {
      for (node_impl &NodeB : Graph.roots()) {
        if (NodeA.isSimilar(NodeB)) {
          if (checkNodeRecursive(NodeA, NodeB)) {
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

  /// Returns the number of nodes in the Graph
  /// @return Number of nodes in the Graph
  size_t getNumberOfNodes() const { return MNodeStorage.size(); }

  /// Traverse the graph recursively to get the events associated with the
  /// output nodes of this graph associated with a specific queue.
  /// @param[in] Queue The queue exit nodes must have been recorded from.
  /// @return vector of events associated to exit nodes.
  std::vector<sycl::detail::EventImplPtr>
  getExitNodesEvents(std::weak_ptr<sycl::detail::queue_impl> Queue);

  /// Sets the Queue state to queue_state::recording. Adds the queue to the list
  /// of recording queues associated with this graph.
  /// Does not take the queue submission lock.
  ///
  /// Required for the cases, when the recording is started directly
  /// from within the kernel submission flow.
  /// @param[in] Queue The queue to be recorded from.
  void beginRecordingUnlockedQueue(sycl::detail::queue_impl &Queue);

  /// Sets the Queue state to queue_state::recording. Adds the queue to the list
  /// of recording queues associated with this graph.
  /// @param[in] Queue The queue to be recorded from.
  void beginRecording(sycl::detail::queue_impl &Queue);

  /// Store the last barrier node that was submitted to the queue.
  /// @param[in] Queue The queue the barrier was recorded from.
  /// @param[in] BarrierNodeImpl The created barrier node.
  void setBarrierDep(std::weak_ptr<sycl::detail::queue_impl> Queue,
                     node_impl &BarrierNodeImpl) {
    MBarrierDependencyMap[Queue] = &BarrierNodeImpl;
  }

  /// Get the last barrier node that was submitted to the queue.
  /// @param[in] Queue The queue to find the last barrier node of. An empty
  /// shared_ptr is returned if no barrier node has been recorded to the queue.
  node_impl *getBarrierDep(std::weak_ptr<sycl::detail::queue_impl> Queue) {
    return MBarrierDependencyMap[Queue];
  }

  unsigned long long getID() const { return MID; }

  /// Get the memory pool used for graph-owned allocations.
  graph_mem_pool &getMemPool() { return MGraphMemPool; }

  /// Mark that an executable graph was created from this modifiable graph, used
  /// for tracking live graphs for graph-owned allocations.
  void markExecGraphCreated() { MExecGraphCount++; }

  /// Mark that an executable graph created from this modifiable graph was
  /// destroyed, used for tracking live graphs for graph-owned allocations.
  void markExecGraphDestroyed() {
    while (true) {
      size_t CurrentVal = MExecGraphCount;
      if (CurrentVal == 0) {
        break;
      }
      if (MExecGraphCount.compare_exchange_strong(CurrentVal, CurrentVal - 1) ==
          false) {
        continue;
      }
    }
  }

  /// Get the number of unique executable graph instances currently alive for
  /// this graph.
  size_t getExecGraphCount() const { return MExecGraphCount; }

  /// Resets the visited edges variable across all nodes in the graph to 0.
  void resetNodeVisitedEdges() {
    for (auto &Node : MNodeStorage) {
      Node->MTotalVisitedEdges = 0;
    }
  }

private:
  template <typename... Ts> node_impl &createNode(Ts &&...Args) {
    MNodeStorage.push_back(
        std::make_shared<node_impl>(std::forward<Ts>(Args)...));
    return *MNodeStorage.back();
  }

  /// Check the graph for cycles by performing a depth-first search of the
  /// graph. If a node is visited more than once in a given path through the
  /// graph, a cycle is present and the search ends immediately.
  /// @return True if a cycle is detected, false if not.
  bool checkForCycles();

  /// Insert node into list of root nodes.
  /// @param Root Node to add to list of root nodes.
  void addRoot(node_impl &Root);

  /// Adds dependencies for a new node, if it has no deps it will be
  /// added as a root node.
  /// @param Node The node to add deps for
  /// @param Deps List of dependent nodes
  void addDepsToNode(node_impl &Node, nodes_range Deps) {
    for (node_impl &N : Deps) {
      N.registerSuccessor(Node);
      this->removeRoot(Node);
    }
    if (Node.MPredecessors.empty()) {
      this->addRoot(Node);
    }
  }

  /// Context associated with this graph.
  sycl::context MContext;
  /// Device associated with this graph. All graph nodes will execute on this
  /// device.
  sycl::device MDevice;

  using RecQueuesStorage =
      std::set<std::weak_ptr<sycl::detail::queue_impl>,
               std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>;
  /// Unique set of queues which are currently recording to this graph.
  RecQueuesStorage MRecordingQueues;
  /// Map of events to their associated recorded nodes.
  std::unordered_map<std::shared_ptr<sycl::detail::event_impl>, node_impl *>
      MEventsMap;
  /// Map for every in-order queue thats recorded a node to the graph, what
  /// the last node added was. We can use this to create new edges on the last
  /// node if any more nodes are added to the graph from the queue.
  std::map<std::weak_ptr<sycl::detail::queue_impl>, node_impl *,
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

  /// Mapping from queues to barrier nodes. For each queue the last barrier
  /// node recorded to the graph from the queue is stored.
  std::map<std::weak_ptr<sycl::detail::queue_impl>, node_impl *,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MBarrierDependencyMap;
  /// Graph memory pool for handling graph-owned memory allocations for this
  /// graph.
  graph_mem_pool MGraphMemPool;

  unsigned long long MID;
  // Used for std::hash in order to create a unique hash for the instance.
  inline static std::atomic<unsigned long long> NextAvailableID = 0;

  // The number of live executable graphs that have been created from this
  // modifiable graph
  std::atomic<size_t> MExecGraphCount = 0;
};

/// Class representing the implementation of command_graph<executable>.
class exec_graph_impl {
public:
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

  /// Constructor.
  ///
  /// Nodes from GraphImpl will be copied when constructing this
  /// exec_graph_impl so that nodes may be modified (e.g. when merging subgraph
  /// nodes).
  /// @param Context Context to create graph with.
  /// @param GraphImpl Modifiable graph implementation to create with.
  /// @param PropList List of properties for constructing this object
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl,
                  const property_list &PropList);

  /// Destructor.
  ///
  /// Releases any UR command-buffers the object has created.
  ~exec_graph_impl();

  /// Partition the graph nodes and put the partition in MPartitions.
  /// The partitioning splits the graph to allow synchronization between
  /// device events and events that do not run on the same device such as
  /// host_task.
  void makePartitions();

  /// Called by handler::ext_oneapi_command_graph() to schedule graph for
  /// execution.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data provided by the sycl::handler
  /// @param EventNeeded Whether an event signalling the completion of this
  /// operation needs to be returned.
  /// @return Returns an event if EventNeeded is true. Returns nullptr
  /// otherwise.
  EventImplPtr enqueue(sycl::detail::queue_impl &Queue,
                       sycl::detail::CG::StorageInitHelper CGData,
                       bool EventNeeded);

  /// Iterates through all the nodes in the graph to build the list of
  /// accessor requirements for the whole graph and for each partition.
  void buildRequirements();

  /// Turns the internal graph representation into UR command-buffers for a
  /// device.
  /// @param Device Device to create backend command-buffers for.
  /// @param Partion Partition to which the created command-buffer should be
  /// attached.
  void createCommandBuffers(sycl::device Device,
                            std::shared_ptr<partition> &Partition);

  /// Query for the device tied to this graph.
  /// @return Device associated with graph.
  sycl::device getDevice() const { return MDevice; }

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

  /// Query the scheduling of node execution.
  /// @return List of nodes in execution order.
  const std::list<node_impl *> &getSchedule() const { return MSchedule; }

  /// Query the graph_impl.
  /// @return pointer to the graph_impl MGraphImpl
  const std::shared_ptr<graph_impl> &getGraphImpl() const { return MGraphImpl; }

  /// Query the vector of the partitions composing the exec_graph.
  /// @return Vector of partitions in execution order.
  const std::vector<std::shared_ptr<partition>> &getPartitions() const {
    return MPartitions;
  }

  nodes_range nodes() const { return MNodeStorage; }

  /// Query whether the graph contains any host-task nodes.
  /// @return True if the graph contains any host-task nodes. False otherwise.
  bool containsHostTask() const { return MContainsHostTask; }

  /// Checks if the previous submissions of this graph have been completed
  /// This function checks the status of events associated to the previous graph
  /// submissions.
  /// @return true if all previous submissions have been completed, false
  /// otherwise.
  bool previousSubmissionCompleted() const {
    for (auto Event : MSchedulerDependencies) {
      if (!Event->isCompleted()) {
        return false;
      }
    }
    return true;
  }

  /// Returns a list of all the accessor requirements for this graph.
  std::vector<sycl::detail::AccessorImplHost *> getRequirements() const {
    return MRequirements;
  }

  void update(std::shared_ptr<graph_impl> GraphImpl);
  void update(node_impl &Node);
  void update(nodes_range Nodes);

  /// Calls UR entry-point to update nodes in command-buffer.
  /// @param CommandBuffer The UR command-buffer to update commands in.
  /// @param Nodes List of nodes to update. Only nodes which can be updated
  /// through UR should be included in this list, currently this is only
  /// nodes of kernel type.
  void updateURImpl(ur_exp_command_buffer_handle_t CommandBuffer,
                    nodes_range Nodes) const;

  /// Update host-task nodes
  /// @param Nodes List of nodes to update, any node that is not a host-task
  /// will be ignored.
  void updateHostTasksImpl(nodes_range Nodes) const;

  /// Splits a list of nodes into separate lists of nodes for each
  /// command-buffer partition.
  ///
  /// Only nodes that can be updated through the UR interface are included
  /// in the list. Currently this is only kernel node types.
  ///
  /// @param Nodes List of nodes to split
  /// @return Map of partition indexes to nodes
  std::map<int, std::vector<node_impl *>>
  getURUpdatableNodes(nodes_range Nodes) const;

  unsigned long long getID() const { return MID; }

  /// Do any work required during finalization to finalize graph-owned memory
  /// allocations.
  void finalizeMemoryAllocations() {
    // This call allocates physical memory and maps all virtual device
    // allocations
    MGraphImpl->getMemPool().allocateAndMapAll();
  }

private:
  // Test helper class for inspecting private graph internals to validate
  // under-the-hood behavior and optimizations.
  friend class ::GraphImplTest;

  /// Create a command-group for the node and add it to command-buffer by going
  /// through the scheduler.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @param IsInOrderPartition True if the partition associated with the node
  /// is a linear (in-order) graph.
  /// @return Optional UR sync point created for this node in the
  /// command-buffer. std::nullopt is returned only if the associated partition
  /// of the node is linear.
  std::optional<ur_exp_command_buffer_sync_point_t>
  enqueueNode(ur_exp_command_buffer_handle_t CommandBuffer, node_impl &Node,
              bool IsInOrderPartition);

  /// Enqueue a node directly to the command-buffer without going through the
  /// scheduler.
  /// @param Ctx Context to use.
  /// @param DeviceImpl Device associated with the enqueue.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @param IsInOrderPartition True if the partition associated with the node
  /// is a linear (in-order) graph.
  /// @return Optional UR sync point created for this node in the
  /// command-buffer. std::nullopt is returned only if the associated partition
  /// of the node is linear.
  std::optional<ur_exp_command_buffer_sync_point_t>
  enqueueNodeDirect(const sycl::context &Ctx,
                    sycl::detail::device_impl &DeviceImpl,
                    ur_exp_command_buffer_handle_t CommandBuffer,
                    node_impl &Node, bool IsInOrderPartition);

  /// Enqueues a host-task partition (i.e. a partition that contains only a
  /// single node and that node is a host-task).
  /// @param Partition The partition to enqueue.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data used for initializing the host-task
  /// command-group.
  /// @param EventNeeded Whether an event signalling the completion of this
  /// operation needs to be returned.
  /// @return If EventNeeded is true returns the event resulting from enqueueing
  /// the host-task through the scheduler. Returns nullptr otherwise.
  EventImplPtr enqueueHostTaskPartition(
      std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
      sycl::detail::CG::StorageInitHelper CGData, bool EventNeeded);

  /// Enqueues a graph partition that contains no host-tasks using the
  /// scheduler.
  /// @param Partition The partition to enqueue.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data used for initializing the command-buffer
  /// command-group.
  /// @param EventNeeded Whether an event signalling the completion of this
  /// operation needs to be returned.
  /// @return If EventNeeded is true returns the event resulting from enqueueing
  /// the command-buffer through the scheduler. Returns nullptr otherwise.
  EventImplPtr enqueuePartitionWithScheduler(
      std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
      sycl::detail::CG::StorageInitHelper CGData, bool EventNeeded);

  /// Enqueues a graph partition that contains no host-tasks by directly calling
  /// the unified-runtime API (i.e. avoids scheduler overhead).
  /// @param Partition The partition to enqueue.
  /// @param Queue Command-queue to schedule execution on.
  /// @param WaitEvents List of events to wait on. All the events on this list
  /// must be safe for scheduler bypass. Only events containing a valid UR event
  /// handle will be waited for.
  /// @param EventNeeded Whether an event signalling the completion of this
  /// operation needs to be returned.
  /// @return If EventNeeded is true returns the event resulting from enqueueing
  /// the command-buffer. Returns nullptr otherwise.
  EventImplPtr enqueuePartitionDirectly(
      std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
      std::vector<detail::EventImplPtr> &WaitEvents, bool EventNeeded);

  /// Enqueues all the partitions in a graph.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data that contains the dependencies and
  /// accessor requirements needed to enqueue this graph.
  /// @param IsCGDataSafeForSchedulerBypass Whether CGData contains any events
  /// that require enqueuing through the scheduler (e.g. requirements or
  /// host-task events).
  /// @param EventNeeded Whether an event signalling the completion of this
  /// operation needs to be returned.
  /// @return If EventNeeded is true returns the event resulting from enqueueing
  /// the command-buffer. Returns nullptr otherwise.
  EventImplPtr enqueuePartitions(sycl::detail::queue_impl &Queue,
                                 sycl::detail::CG::StorageInitHelper &CGData,
                                 bool IsCGDataSafeForSchedulerBypass,
                                 bool EventNeeded);

  /// Iterates back through predecessors to find the real dependency.
  /// @param[out] Deps Found dependencies.
  /// @param[in] CurrentNode Node to find dependencies for.
  /// @param[in] ReferencePartitionNum Number of the partition containing the
  /// SyncPoint for CurrentNode, otherwise we need to
  /// synchronize on the host with the completion of previous partitions.
  void findRealDeps(std::vector<ur_exp_command_buffer_sync_point_t> &Deps,
                    node_impl &CurrentNode, int ReferencePartitionNum);

  /// Duplicate nodes from the modifiable graph associated with this executable
  /// graph and store them locally. Any subgraph nodes in the modifiable graph
  /// will be expanded and merged into this new set of nodes.
  void duplicateNodes();

  /// Prints the contents of the graph to a text file in DOT format.
  /// @param FilePath Path to the output file.
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printGraphAsDot(const std::string FilePath, bool Verbose) const {
    /// Vector of nodes visited during the graph printing
    std::vector<node_impl *> VisitedNodes;

    std::fstream Stream(FilePath, std::ios::out);
    Stream << "digraph dot {" << std::endl;

    std::vector<node_impl *> Roots;
    for (node_impl &Node : nodes()) {
      if (Node.MPredecessors.size() == 0) {
        Roots.push_back(&Node);
      }
    }

    for (node_impl *Node : Roots)
      Node->printDotRecursive(Stream, VisitedNodes, Verbose);

    Stream << "}" << std::endl;

    Stream.close();
  }

  /// Determines if scheduler needs to be used for node update.
  /// @param[in] Nodes List of nodes to be updated
  /// @param[out] UpdateRequirements Accessor requirements found in /p Nodes.
  /// return True if update should be done through the scheduler.
  bool needsScheduledUpdate(
      nodes_range Nodes,
      std::vector<sycl::detail::AccessorImplHost *> &UpdateRequirements);

  /// Sets the UR struct values required to update a graph node.
  /// @param[in] Node The node to be updated.
  /// @param[out] BundleObjs UR objects created from kernel bundle.
  /// Responsibility of the caller to release.
  /// @param[out] MemobjDescs Memory object arguments to update.
  /// @param[out] MemobjProps Properties used in /p MemobjDescs structs.
  /// @param[out] PtrDescs Pointer arguments to update.
  /// @param[out] ValueDescs Value arguments to update.
  /// @param[out] NDRDesc ND-Range to update.
  /// @param[out] UpdateDesc Base struct in the pointer chain.
  void populateURKernelUpdateStructs(
      node_impl &Node, FastKernelCacheValPtr &BundleObjs,
      std::vector<ur_exp_command_buffer_update_memobj_arg_desc_t> &MemobjDescs,
      std::vector<ur_kernel_arg_mem_obj_properties_t> &MemobjProps,
      std::vector<ur_exp_command_buffer_update_pointer_arg_desc_t> &PtrDescs,
      std::vector<ur_exp_command_buffer_update_value_arg_desc_t> &ValueDescs,
      sycl::detail::NDRDescT &NDRDesc,
      ur_exp_command_buffer_update_kernel_launch_desc_t &UpdateDesc) const;

  /// Execution schedule of nodes in the graph.
  std::list<node_impl *> MSchedule;
  /// Pointer to the modifiable graph impl associated with this executable
  /// graph.
  /// Thread-safe implementation note: in the current implementation
  /// multiple exec_graph_impl can reference the same graph_impl object.
  /// This specificity must be taken into account when trying to lock
  /// the graph_impl mutex from an exec_graph_impl to avoid deadlock.
  std::shared_ptr<graph_impl> MGraphImpl;
  /// Map of nodes in the exec graph to the sync point representing their
  /// execution in the command graph.
  std::unordered_map<node_impl *, ur_exp_command_buffer_sync_point_t>
      MSyncPoints;
  /// Sycl queue impl ptr associated with this graph.
  std::shared_ptr<sycl::detail::queue_impl> MQueueImpl;
  /// Map of nodes in the exec graph to the partition number to which they
  /// belong.
  std::unordered_map<node_impl *, int> MPartitionNodes;
  /// Device associated with this executable graph.
  sycl::device MDevice;
  /// Context associated with this executable graph.
  sycl::context MContext;
  /// List of requirements for enqueueing this command graph, accumulated from
  /// all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;
  /// List of dependencies that enqueue or update commands need to wait on
  /// when using the scheduler path.
  std::vector<sycl::detail::EventImplPtr> MSchedulerDependencies;
  /// List of the partitions that compose the exec graph.
  std::vector<std::shared_ptr<partition>> MPartitions;
  /// Storage for copies of nodes from the original modifiable graph.
  std::vector<std::shared_ptr<node_impl>> MNodeStorage;
  /// Map of nodes to their associated UR command handles.
  std::unordered_map<node_impl *, ur_exp_command_buffer_command_handle_t>
      MCommandMap;
  /// List of partition without any predecessors in this exec graph.
  std::vector<std::weak_ptr<partition>> MRootPartitions;
  /// True if this graph can be updated (set with property::updatable)
  bool MIsUpdatable;
  /// If true, the graph profiling is enabled.
  bool MEnableProfiling;

  // Stores a cache of node ids from modifiable graph nodes to the companion
  // node(s) in this graph. Used for quick access when updating this graph.
  std::multimap<node_impl::id_type, node_impl *> MIDCache;

  unsigned long long MID;
  // Used for std::hash in order to create a unique hash for the instance.
  inline static std::atomic<unsigned long long> NextAvailableID = 0;

  // True if this graph contains any host-tasks, indicates we need special
  // handling for them during update().
  bool MContainsHostTask = false;
};
} // namespace detail
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
