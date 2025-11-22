//==--------- graph_impl.cpp - SYCL graph extension ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SYCL_GRAPH_IMPL_CPP

#include "graph_impl.hpp"
#include "dynamic_impl.hpp" // for dynamic classes
#include "node_impl.hpp"    // for node_impl
#include <detail/cg.hpp> // for CG, CGExecKernel, CGHostTask, ArgDesc, NDRDescT
#include <detail/event_impl.hpp>                      // for event_impl
#include <detail/handler_impl.hpp>                    // for handler_impl
#include <detail/kernel_arg_mask.hpp>                 // for KernelArgMask
#include <detail/kernel_impl.hpp>                     // for kernel_impl
#include <detail/program_manager/program_manager.hpp> // ProgramManager
#include <detail/queue_impl.hpp>                      // for queue_impl
#include <detail/sycl_mem_obj_t.hpp>                  // for SYCLMemObjT
#include <stack>                                      // for stack
#include <sycl/detail/common.hpp>      // for tls_code_loc_t etc..
#include <sycl/detail/kernel_desc.hpp> // for kernel_param_kind_t
#include <sycl/detail/string_view.hpp> // for string_view
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.hpp> // for checking graph properties
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp> // for command_graph
#include <sycl/ext/oneapi/experimental/graph/common.hpp> // for graph_state
#include <sycl/ext/oneapi/experimental/graph/executable_graph.hpp> // for executable_command_graph
#include <sycl/ext/oneapi/experimental/graph/modifiable_graph.hpp> // for modifiable_command_graph
#include <sycl/feature_test.hpp> // for testing
#include <sycl/queue.hpp>        // for queue

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

namespace {
/// Return a string representation of a given node_type
inline const char *nodeTypeToString(node_type NodeType) {
  switch (NodeType) {
  case node_type::empty:
    return "empty";
  case node_type::subgraph:
    return "subgraph";
  case node_type::kernel:
    return "kernel";
  case node_type::memcpy:
    return "memcpy";
  case node_type::memset:
    return "memset";
  case node_type::memfill:
    return "memfill";
  case node_type::prefetch:
    return "prefetch";
  case node_type::memadvise:
    return "memadvise";
  case node_type::ext_oneapi_barrier:
    return "ext_oneapi_barrier";
  case node_type::host_task:
    return "host_task";
  case node_type::native_command:
    return "native_command";
  case node_type::async_malloc:
    return "async_malloc";
  case node_type::async_free:
    return "async_free";
  }
  assert(false && "Unhandled node type");
  return {};
}

/// Topologically sorts the graph in order to schedule nodes for execution.
/// This implementation is based on Kahn's algorithm which uses a Breadth-first
/// search approach.
/// For performance reasons, this function uses the MTotalVisitedEdges
/// member variable of the node_impl class. It's the caller responsibility to
/// make sure that MTotalVisitedEdges is set to 0 for all nodes in the graph
/// before calling this function.
/// @param[in] Roots List of root nodes.
/// @param[out] SortedNodes The graph nodes sorted in topological order.
/// @param[in] PartitionBounded If set to true, the topological sort is stopped
/// at partition borders. Hence, nodes belonging to a partition different from
/// the NodeImpl partition are not processed.
void sortTopological(nodes_range Roots, std::list<node_impl *> &SortedNodes,
                     bool PartitionBounded) {
  std::stack<node_impl *> Source;

  for (node_impl &Node : Roots) {
    Source.push(&Node);
  }

  while (!Source.empty()) {
    node_impl &Node = *Source.top();
    Source.pop();
    SortedNodes.push_back(&Node);

    for (node_impl &Succ : Node.successors()) {

      if (PartitionBounded && (Succ.MPartitionNum != Node.MPartitionNum)) {
        continue;
      }

      auto &TotalVisitedEdges = Succ.MTotalVisitedEdges;
      ++TotalVisitedEdges;
      if (TotalVisitedEdges == Succ.MPredecessors.size()) {
        Source.push(&Succ);
      }
    }
  }
}

/// Propagates the partition number `PartitionNum` to predecessors.
/// Propagation stops when a host task is encountered or when no predecessors
/// remain or when we encounter a node that has already been processed and has a
/// partition number lower that the one propagated here. Indeed,
/// partition numbers reflect the execution order. Hence, the partition number
/// of a node can be decreased but not increased. Moreover, as predecessors of a
/// node are either in the same partition or a partition with a smaller number,
/// we do not need to continue propagating the partition number if we encounter
/// a node with a smaller partition number.
/// @param Node Node to assign to the partition.
/// @param PartitionNum Number to propagate.
void propagatePartitionUp(node_impl &Node, int PartitionNum) {
  if (((Node.MPartitionNum != -1) && (Node.MPartitionNum <= PartitionNum)) ||
      (Node.MCGType == sycl::detail::CGType::CodeplayHostTask)) {
    return;
  }
  Node.MPartitionNum = PartitionNum;
  for (node_impl &Predecessor : Node.predecessors()) {
    propagatePartitionUp(Predecessor, PartitionNum);
  }
}

/// Propagates the partition number `PartitionNum` to successors.
/// Propagation stops when an host task is encountered or when no successors
/// remain.
/// @param Node Node to assign to the partition.
/// @param PartitionNum Number to propagate.
/// @param HostTaskList List of host tasks that have already been processed and
/// are encountered as successors to the node Node.
void propagatePartitionDown(node_impl &Node, int PartitionNum,
                            std::list<node_impl *> &HostTaskList) {
  if (Node.MCGType == sycl::detail::CGType::CodeplayHostTask) {
    if (Node.MPartitionNum != -1) {
      HostTaskList.push_front(&Node);
    }
    return;
  }
  Node.MPartitionNum = PartitionNum;
  for (node_impl &Successor : Node.successors()) {
    propagatePartitionDown(Successor, PartitionNum, HostTaskList);
  }
}

/// Tests if the node is a root of its partition (i.e. no predecessors that
/// belong to the same partition)
/// @param Node node to test
/// @return True is `Node` is a root of its partition
bool isPartitionRoot(node_impl &Node) {
  for (node_impl &Predecessor : Node.predecessors()) {
    if (Predecessor.MPartitionNum == Node.MPartitionNum) {
      return false;
    }
  }
  return true;
}
} // anonymous namespace

void partition::updateSchedule() {
  if (MSchedule.empty()) {
    // There is no need to reset MTotalVisitedEdges before calling
    // sortTopological because this function is only called once per partition.
    sortTopological(MRoots, MSchedule, true);
  }
}

void exec_graph_impl::makePartitions() {
  int CurrentPartition = -1;
  std::list<node_impl *> HostTaskList;
  // find all the host-tasks in the graph
  for (node_impl &Node : nodes()) {
    if (Node.MCGType == sycl::detail::CGType::CodeplayHostTask) {
      HostTaskList.push_back(&Node);
    }
  }

  MContainsHostTask = HostTaskList.size() > 0;
  // Annotate nodes
  // The first step in graph partitioning is to annotate all nodes of the graph
  // with a temporary partition or group number. This step allows us to group
  // the graph nodes into sets of nodes with kind of meta-dependencies that must
  // be enforced by the runtime. For example, Group 2 depends on Groups 0 and 1,
  // which means that we should not try to run Group 2 before Groups 0 and 1
  // have finished executing. Since host-tasks are currently the only tasks that
  // require runtime dependency handling, groups of nodes are created from
  // host-task nodes. We therefore loop over all the host-task nodes, and for
  // each node:
  //  - Its predecessors are assigned to group number `n-1`
  //  - The node itself constitutes a group, group number `n`
  //  - Its successors are assigned to group number `n+1`
  // Since running multiple partitions slows down the whole graph execution, we
  // then try to reduce the number of partitions by merging them when possible.
  // Typically, the grouping algorithm can create two successive partitions
  // of target nodes in the following case:
  // A host-task `A` is added to the graph. Later, another host task `B` is
  // added to the graph. Consequently, the node `A` is stored before the node
  // `B` in the node storage vector. Now, if `A` is placed as a successor of `B`
  // (using make_edge function to make node `A` dependent on node `B`.) In this
  // case, the host-task node `A` must be reprocessed after the node `B` and the
  // group that includes the predecessor of `B` can be merged with the group of
  // the predecessors of the node `A`.
  while (HostTaskList.size() > 0) {
    node_impl &Node = *HostTaskList.front();
    HostTaskList.pop_front();
    CurrentPartition++;
    for (node_impl &Predecessor : Node.predecessors()) {
      propagatePartitionUp(Predecessor, CurrentPartition);
    }
    CurrentPartition++;
    Node.MPartitionNum = CurrentPartition;
    CurrentPartition++;
    auto TmpSize = HostTaskList.size();
    for (node_impl &Successor : Node.successors()) {
      propagatePartitionDown(Successor, CurrentPartition, HostTaskList);
    }
    if (HostTaskList.size() > TmpSize) {
      // At least one HostTask has been re-numbered so group merge opportunities
      for (node_impl *HT : HostTaskList) {
        auto HTPartitionNum = HT->MPartitionNum;
        if (HTPartitionNum != -1) {
          // can merge predecessors of node `Node` with predecessors of node
          // `HT` (HTPartitionNum-1) since HT must be reprocessed
          for (node_impl &NodeImpl : nodes()) {
            if (NodeImpl.MPartitionNum == Node.MPartitionNum - 1) {
              NodeImpl.MPartitionNum = HTPartitionNum - 1;
            }
          }
        } else {
          break;
        }
      }
    }
  }

  // Create partitions
  int PartitionFinalNum = 0;
  for (int i = -1; i <= CurrentPartition; i++) {
    const std::shared_ptr<partition> &Partition = std::make_shared<partition>();
    for (node_impl &Node : nodes()) {
      if (Node.MPartitionNum == i) {
        MPartitionNodes[&Node] = PartitionFinalNum;
        if (isPartitionRoot(Node)) {
          Partition->MRoots.insert(&Node);
          if (Node.MCGType == CGType::CodeplayHostTask) {
            Partition->MIsHostTask = true;
          }
        }
      }
    }
    if (Partition->MRoots.size() > 0) {
      Partition->updateSchedule();
      Partition->MIsInOrderGraph = Partition->checkIfGraphIsSinglePath();
      MPartitions.push_back(Partition);
      MRootPartitions.push_back(Partition);
      PartitionFinalNum++;
    }
  }

  // Add an empty partition if there is no partition, i.e. empty graph
  if (MPartitions.empty()) {
    MPartitions.push_back(std::make_shared<partition>());
    MRootPartitions.push_back(MPartitions[0]);
  }

  // Make global schedule list
  for (const auto &Partition : MPartitions) {
    MSchedule.insert(MSchedule.end(), Partition->MSchedule.begin(),
                     Partition->MSchedule.end());
  }

  // Compute partition dependencies
  for (const auto &Partition : MPartitions) {
    for (node_impl &Root : Partition->roots()) {
      for (node_impl &NodeDep : Root.predecessors()) {
        auto &Predecessor = MPartitions[MPartitionNodes[&NodeDep]];
        Partition->MPredecessors.push_back(Predecessor.get());
        Predecessor->MSuccessors.push_back(Partition.get());
      }
    }
  }

  // Reset node groups (if node have to be re-processed - e.g. subgraph)
  for (node_impl &Node : nodes()) {
    Node.MPartitionNum = -1;
  }
}

graph_impl::graph_impl(const sycl::context &SyclContext,
                       const sycl::device &SyclDevice,
                       const sycl::property_list &PropList)
    : MContext(SyclContext), MDevice(SyclDevice), MRecordingQueues(),
      MEventsMap(), MInorderQueueMap(),
      MGraphMemPool(*this, SyclContext, SyclDevice),
      MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {
  checkGraphPropertiesAndThrow(PropList);
  if (PropList.has_property<property::graph::no_cycle_check>()) {
    MSkipCycleChecks = true;
  }
  if (PropList.has_property<property::graph::assume_buffer_outlives_graph>()) {
    MAllowBuffers = true;
  }

  if (!SyclDevice.has(aspect::ext_oneapi_limited_graph) &&
      !SyclDevice.has(aspect::ext_oneapi_graph)) {
    std::stringstream Stream;
    Stream << SyclDevice.get_backend();
    std::string BackendString = Stream.str();
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        BackendString + " backend is not supported by SYCL Graph extension.");
  }
}

graph_impl::~graph_impl() {
  try {
    clearQueues(false /*Needs lock*/);
    for (auto &MemObj : MMemObjs) {
      MemObj->markNoLongerBeingUsedInGraph();
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~graph_impl", e);
  }
}

void graph_impl::addRoot(node_impl &Root) { MRoots.insert(&Root); }

void graph_impl::removeRoot(node_impl &Root) { MRoots.erase(&Root); }

std::set<node_impl *> graph_impl::getCGEdges(
    const std::shared_ptr<sycl::detail::CG> &CommandGroup) const {
  const auto &Requirements = CommandGroup->getRequirements();
  if (!MAllowBuffers && Requirements.size()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Cannot use buffers in a graph without passing the "
                          "assume_buffer_outlives_graph property on "
                          "Graph construction.");
  }

  if (CommandGroup->getType() == sycl::detail::CGType::Kernel) {
    auto CGKernel =
        static_cast<sycl::detail::CGExecKernel *>(CommandGroup.get());
    if (CGKernel->hasStreams()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Using sycl streams in a graph node is unsupported.");
    }
  }

  // Add any nodes specified by event dependencies into the dependency list
  std::set<node_impl *> UniqueDeps;
  for (auto &Dep : CommandGroup->getEvents()) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl == MEventsMap.end()) {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    } else {
      UniqueDeps.insert(NodeImpl->second);
    }
  }

  // A unique set of dependencies obtained by checking requirements and events
  for (auto &Req : Requirements) {
    // Look through the graph for nodes which share this requirement
    for (node_impl &Node : nodes()) {
      if (Node.hasRequirementDependency(Req) &&
          // If any of this node's successors have this requirement then we skip
          // adding the current node as a dependency.
          none_of(Node.successors(), [&](node_impl &Succ) {
            return Succ.hasRequirementDependency(Req);
          })) {
        UniqueDeps.insert(&Node);
      }
    }
  }

  return UniqueDeps;
}

void graph_impl::markCGMemObjs(
    const std::shared_ptr<sycl::detail::CG> &CommandGroup) {
  const auto &Requirements = CommandGroup->getRequirements();
  for (auto &Req : Requirements) {
    auto MemObj = static_cast<sycl::detail::SYCLMemObjT *>(Req->MSYCLMemObj);
    bool WasInserted = MMemObjs.insert(MemObj).second;
    if (WasInserted) {
      MemObj->markBeingUsedInGraph();
    }
  }
}

node_impl &graph_impl::add(nodes_range Deps) {
  node_impl &NodeImpl = createNode();

  addDepsToNode(NodeImpl, Deps);
  // Add an event associated with this explicit node for mixed usage
  addEventForNode(sycl::detail::event_impl::create_completed_host_event(),
                  NodeImpl);
  return NodeImpl;
}

node_impl &graph_impl::add(std::function<void(handler &)> CGF,
                           const std::vector<sycl::detail::ArgDesc> &Args,
                           nodes_range Deps) {
  (void)Args;
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  detail::handler_impl HandlerImpl{*this};
  sycl::handler Handler{HandlerImpl};
#else
  sycl::handler Handler{shared_from_this()};
#endif

  // Pass the node deps to the handler so they are available when processing the
  // CGF, need for async_malloc nodes.
  for (node_impl &N : Deps)
    Handler.impl->MNodeDeps.push_back(N.shared_from_this());

#if XPTI_ENABLE_INSTRUMENTATION
  // Save code location if one was set in TLS.
  // Ideally it would be nice to capture user's call code location
  // by adding a parameter to the graph.add function, but this will
  // break the API. At least capture code location from TLS, user
  // can set it before calling graph.add
  if (xptiTraceEnabled()) {
    sycl::detail::tls_code_loc_t Tls;
    Handler.saveCodeLoc(Tls.query(), Tls.isToplevel());
  }
#endif

  CGF(Handler);

  if (Handler.getType() == sycl::detail::CGType::Barrier) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "The sycl_ext_oneapi_enqueue_barrier feature is not available with "
        "SYCL Graph Explicit API. Please use empty nodes instead.");
  }

  Handler.finalize();

  // In explicit mode the handler processing of the CGF does not need a write
  // lock as it does not modify the graph, we extract information from it here
  // and modify the graph.
  graph_impl::WriteLock Lock(MMutex);
  node_type NodeType =
      Handler.impl->MUserFacingNodeType !=
              ext::oneapi::experimental::node_type::empty
          ? Handler.impl->MUserFacingNodeType
          : ext::oneapi::experimental::detail::getNodeTypeFromCG(
                Handler.getType());

  node_impl &NodeImpl =
      this->add(NodeType, std::move(Handler.impl->MGraphNodeCG), Deps);

  // Add an event associated with this explicit node for mixed usage
  addEventForNode(sycl::detail::event_impl::create_completed_host_event(),
                  NodeImpl);

  // Retrieve any dynamic parameters which have been registered in the CGF and
  // register the actual nodes with them.
  auto &DynamicParams = Handler.impl->MKernelData.getDynamicParameters();

  if (NodeType != node_type::kernel && DynamicParams.size() > 0) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "dynamic_parameters cannot be registered with graph "
                          "nodes which do not represent kernel executions");
  }

  for (auto &[DynamicParam, ArgIndex] : DynamicParams) {
    DynamicParam->registerNode(NodeImpl, ArgIndex);
  }

  return NodeImpl;
}

node_impl &graph_impl::add(node_type NodeType,
                           std::shared_ptr<sycl::detail::CG> CommandGroup,
                           nodes_range Deps) {

  // A unique set of dependencies obtained by checking requirements and events
  std::set<node_impl *> UniqueDeps = getCGEdges(CommandGroup);

  // Track and mark the memory objects being used by the graph.
  markCGMemObjs(CommandGroup);

  node_impl &NodeImpl = createNode(NodeType, std::move(CommandGroup));

  // Add any deps determined from requirements and events into the dependency
  // list
  addDepsToNode(NodeImpl, Deps);
  addDepsToNode(NodeImpl, UniqueDeps);

  if (NodeType == node_type::async_free) {
    auto AsyncFreeCG = static_cast<CGAsyncFree *>(NodeImpl.MCommandGroup.get());
    // If this is an async free node mark that it is now available for reuse,
    // and pass the async free node for tracking.
    MGraphMemPool.markAllocationAsAvailable(AsyncFreeCG->getPtr(), NodeImpl);
  }

  return NodeImpl;
}

node_impl &
graph_impl::add(std::shared_ptr<dynamic_command_group_impl> &DynCGImpl,
                nodes_range Deps) {
  // Set of Dependent nodes based on CG event and accessor dependencies.
  std::set<node_impl *> DynCGDeps = getCGEdges(DynCGImpl->MCommandGroups[0]);
  for (unsigned i = 1; i < DynCGImpl->getNumCGs(); i++) {
    auto &CG = DynCGImpl->MCommandGroups[i];
    auto CGEdges = getCGEdges(CG);
    if (CGEdges != DynCGDeps) {
      throw sycl::exception(make_error_code(sycl::errc::invalid),
                            "Command-groups in dynamic command-group don't have"
                            "equivalent dependencies to other graph nodes.");
    }
  }

  // Track and mark the memory objects being used by the graph.
  for (auto &CG : DynCGImpl->MCommandGroups) {
    markCGMemObjs(CG);
  }

  // Get active dynamic command-group CG and use to create a node object
  const auto &ActiveKernel = DynCGImpl->getActiveCG();
  node_type NodeType =
      ext::oneapi::experimental::detail::getNodeTypeFromCG(DynCGImpl->MCGType);
  detail::node_impl &NodeImpl = add(NodeType, ActiveKernel, Deps);

  // Add an event associated with this explicit node for mixed usage
  addEventForNode(sycl::detail::event_impl::create_completed_host_event(),
                  NodeImpl);

  // Track the dynamic command-group used inside the node object
  DynCGImpl->MNodes.push_back(NodeImpl.shared_from_this());

  return NodeImpl;
}

std::shared_ptr<sycl::detail::queue_impl> graph_impl::getQueue() const {
  std::shared_ptr<sycl::detail::queue_impl> Return{};
  if (!MRecordingQueues.empty())
    Return = MRecordingQueues.begin()->lock();
  return Return;
}

void graph_impl::addQueue(sycl::detail::queue_impl &RecordingQueue) {
  MRecordingQueues.insert(RecordingQueue.weak_from_this());
}

void graph_impl::removeQueue(sycl::detail::queue_impl &RecordingQueue) {
  MRecordingQueues.erase(RecordingQueue.weak_from_this());
}

void graph_impl::clearQueues(bool NeedsLock) {
  graph_impl::RecQueuesStorage SwappedQueues;
  {
    graph_impl::WriteLock Guard(MMutex, std::defer_lock);
    if (NeedsLock) {
      Guard.lock();
    }
    std::swap(MRecordingQueues, SwappedQueues);
  }

  for (auto &Queue : SwappedQueues) {
    if (auto ValidQueue = Queue.lock(); ValidQueue) {
      ValidQueue->setCommandGraph(nullptr);
    }
  }
}

bool graph_impl::checkForCycles() {
  std::list<node_impl *> SortedNodes;
  sortTopological(MRoots, SortedNodes, false);

  // If after a topological sort, not all the nodes in the graph are sorted,
  // then there must be at least one cycle in the graph. This is guaranteed
  // by Kahn's algorithm, which sortTopological() implements.
  bool CycleFound = SortedNodes.size() != MNodeStorage.size();

  // Reset the MTotalVisitedEdges variable to prepare for the next cycle check.
  for (auto &Node : MNodeStorage) {
    Node->MTotalVisitedEdges = 0;
  }

  return CycleFound;
}

node_impl *graph_impl::getLastInorderNode(sycl::detail::queue_impl *Queue) {
  if (!Queue) {
    assert(0 ==
           MInorderQueueMap.count(std::weak_ptr<sycl::detail::queue_impl>{}));
    return {};
  }
  if (0 == MInorderQueueMap.count(Queue->weak_from_this())) {
    return {};
  }
  return MInorderQueueMap[Queue->weak_from_this()];
}

void graph_impl::setLastInorderNode(sycl::detail::queue_impl &Queue,
                                    node_impl &Node) {
  MInorderQueueMap[Queue.weak_from_this()] = &Node;
}

void graph_impl::makeEdge(node_impl &Src, node_impl &Dest) {
  throwIfGraphRecordingQueue("make_edge()");
  if (&Src == &Dest) {
    throw sycl::exception(
        make_error_code(sycl::errc::invalid),
        "make_edge() cannot be called when Src and Dest are the same.");
  }

  bool SrcFound = false;
  bool DestFound = false;
  for (const auto &Node : MNodeStorage) {

    SrcFound |= Node.get() == &Src;
    DestFound |= Node.get() == &Dest;

    if (SrcFound && DestFound) {
      break;
    }
  }

  if (!SrcFound) {
    throw sycl::exception(make_error_code(sycl::errc::invalid),
                          "Src must be a node inside the graph.");
  }
  if (!DestFound) {
    throw sycl::exception(make_error_code(sycl::errc::invalid),
                          "Dest must be a node inside the graph.");
  }

  bool DestWasGraphRoot = Dest.MPredecessors.size() == 0;

  // We need to add the edges first before checking for cycles
  Src.registerSuccessor(Dest);

  bool DestLostRootStatus = DestWasGraphRoot && Dest.MPredecessors.size() == 1;
  if (DestLostRootStatus) {
    // Dest is no longer a Root node, so we need to remove it from MRoots.
    MRoots.erase(&Dest);
  }

  // We can skip cycle checks if either Dest has no successors (cycle not
  // possible) or cycle checks have been disabled with the no_cycle_check
  // property;
  if (Dest.MSuccessors.empty() || !MSkipCycleChecks) {
    bool CycleFound = checkForCycles();

    if (CycleFound) {
      // Remove the added successor and predecessor.
      Src.MSuccessors.pop_back();
      Dest.MPredecessors.pop_back();
      if (DestLostRootStatus) {
        // Add Dest back into MRoots.
        MRoots.insert(&Dest);
      }

      throw sycl::exception(make_error_code(sycl::errc::invalid),
                            "Command graphs cannot contain cycles.");
    }
  }
  removeRoot(Dest); // remove receiver from root node list
}

std::vector<sycl::detail::EventImplPtr> graph_impl::getExitNodesEvents(
    std::weak_ptr<sycl::detail::queue_impl> RecordedQueue) {
  std::vector<sycl::detail::EventImplPtr> Events;

  auto RecordedQueueSP = RecordedQueue.lock();
  for (node_impl &Node : nodes()) {
    if (Node.MSuccessors.empty()) {
      auto EventForNode = getEventForNode(Node);
      if (EventForNode->getSubmittedQueue() == RecordedQueueSP) {
        Events.push_back(getEventForNode(Node));
      }
    }
  }

  return Events;
}

void graph_impl::beginRecordingUnlockedQueue(sycl::detail::queue_impl &Queue) {
  graph_impl::WriteLock Lock(MMutex);
  if (!Queue.hasCommandGraph()) {
    Queue.setCommandGraphUnlocked(shared_from_this());
    addQueue(Queue);
  }
}

void graph_impl::beginRecording(sycl::detail::queue_impl &Queue) {
  graph_impl::WriteLock Lock(MMutex);
  if (!Queue.hasCommandGraph()) {
    Queue.setCommandGraph(shared_from_this());
    addQueue(Queue);
  }
}

// Check if nodes do not require enqueueing and if so loop back through
// predecessors until we find the real dependency.
void exec_graph_impl::findRealDeps(
    std::vector<ur_exp_command_buffer_sync_point_t> &Deps,
    node_impl &CurrentNode, int ReferencePartitionNum) {
  if (!CurrentNode.requiresEnqueue()) {
    for (node_impl &NodeImpl : CurrentNode.predecessors()) {
      findRealDeps(Deps, NodeImpl, ReferencePartitionNum);
    }
  } else {
    auto CurrentNodePtr = CurrentNode.shared_from_this();
    // Verify if CurrentNode belong the the same partition
    if (MPartitionNodes[&CurrentNode] == ReferencePartitionNum) {
      // Verify that the sync point has actually been set for this node.
      auto SyncPoint = MSyncPoints.find(&CurrentNode);
      assert(SyncPoint != MSyncPoints.end() &&
             "No sync point has been set for node dependency.");
      // Check if the dependency has already been added.
      if (std::find(Deps.begin(), Deps.end(), SyncPoint->second) ==
          Deps.end()) {
        Deps.push_back(SyncPoint->second);
      }
    }
  }
}

std::optional<ur_exp_command_buffer_sync_point_t>
exec_graph_impl::enqueueNodeDirect(const sycl::context &Ctx,
                                   sycl::detail::device_impl &DeviceImpl,
                                   ur_exp_command_buffer_handle_t CommandBuffer,
                                   node_impl &Node, bool IsInOrderPartition) {
  std::vector<ur_exp_command_buffer_sync_point_t> Deps;
  if (!IsInOrderPartition) {
    for (node_impl &N : Node.predecessors()) {
      findRealDeps(Deps, N, MPartitionNodes[&Node]);
    }
  }
  ur_exp_command_buffer_sync_point_t NewSyncPoint;
  ur_exp_command_buffer_command_handle_t NewCommand = 0;

#ifdef XPTI_ENABLE_INSTRUMENTATION
  const bool xptiEnabled = xptiTraceEnabled();
  xpti_td *CmdTraceEvent = nullptr;
  uint64_t InstanceID = 0;
  uint8_t StreamID = 0;
  if (xptiEnabled) {
    StreamID = detail::getActiveXPTIStreamID();
    sycl::detail::CGExecKernel *CGExec =
        static_cast<sycl::detail::CGExecKernel *>(Node.MCommandGroup.get());
    sycl::detail::code_location CodeLoc(CGExec->MFileName.c_str(),
                                        CGExec->MFunctionName.c_str(),
                                        CGExec->MLine, CGExec->MColumn);
    std::tie(CmdTraceEvent, InstanceID) = emitKernelInstrumentationData(
        StreamID, CGExec->MSyclKernel.get(), CodeLoc, CGExec->MIsTopCodeLoc,
        CGExec->MDeviceKernelInfo, nullptr, CGExec->MNDRDesc,
        CGExec->MKernelBundle.get(), CGExec->MArgs);
    if (CmdTraceEvent)
      sycl::detail::emitInstrumentationGeneral(
          StreamID, InstanceID, CmdTraceEvent, xpti::trace_task_begin, nullptr);
  }
#endif

  ur_result_t Res = sycl::detail::enqueueImpCommandBufferKernel(
      Ctx, DeviceImpl, CommandBuffer,
      *static_cast<sycl::detail::CGExecKernel *>((Node.MCommandGroup.get())),
      Deps, IsInOrderPartition ? nullptr : &NewSyncPoint,
      MIsUpdatable ? &NewCommand : nullptr, nullptr);

  if (MIsUpdatable) {
    MCommandMap[&Node] = NewCommand;
  }

  if (Res != UR_RESULT_SUCCESS) {
    throw sycl::exception(errc::invalid,
                          "Failed to add kernel to UR command-buffer");
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiEnabled && CmdTraceEvent)
    sycl::detail::emitInstrumentationGeneral(
        StreamID, InstanceID, CmdTraceEvent, xpti::trace_task_end, nullptr);
#endif

  // Linear (in-order) graphs do not return a sync point as the dependencies of
  // successor nodes are handled by the UR CommandBuffer via the isInOrder flag
  return IsInOrderPartition
             ? std::nullopt
             : std::optional<ur_exp_command_buffer_sync_point_t>{NewSyncPoint};
}

std::optional<ur_exp_command_buffer_sync_point_t>
exec_graph_impl::enqueueNode(ur_exp_command_buffer_handle_t CommandBuffer,
                             node_impl &Node, bool IsInOrderPartition) {
  std::vector<ur_exp_command_buffer_sync_point_t> Deps;
  if (!IsInOrderPartition) {
    for (node_impl &N : Node.predecessors()) {
      findRealDeps(Deps, N, MPartitionNodes[&Node]);
    }
  }

  sycl::detail::EventImplPtr Event =
      sycl::detail::Scheduler::getInstance().addCG(
          Node.getCGCopy(), *MQueueImpl,
          /*EventNeeded=*/true, CommandBuffer, Deps);

  if (MIsUpdatable) {
    MCommandMap[&Node] = Event->getCommandBufferCommand();
  }

  // Linear (in-order) graphs do not return a sync point as the dependencies of
  // successor nodes are handled by the UR CommandBuffer via the isInOrder flag
  return IsInOrderPartition ? std::nullopt
                            : std::optional<ur_exp_command_buffer_sync_point_t>{
                                  Event->getSyncPoint()};
}

void exec_graph_impl::buildRequirements() {

  for (auto &Node : MNodeStorage) {
    if (!Node->MCommandGroup)
      continue;

    MRequirements.insert(MRequirements.end(),
                         Node->MCommandGroup->getRequirements().begin(),
                         Node->MCommandGroup->getRequirements().end());

    std::shared_ptr<partition> &Partition =
        MPartitions[MPartitionNodes[Node.get()]];

    Partition->MRequirements.insert(
        Partition->MRequirements.end(),
        Node->MCommandGroup->getRequirements().begin(),
        Node->MCommandGroup->getRequirements().end());

    Partition->MAccessors.insert(Partition->MAccessors.end(),
                                 Node->MCommandGroup->getAccStorage().begin(),
                                 Node->MCommandGroup->getAccStorage().end());
  }
}

void exec_graph_impl::createCommandBuffers(
    sycl::device Device, std::shared_ptr<partition> &Partition) {
  const bool IsInOrderCommandBuffer =
      Partition->MIsInOrderGraph && !MEnableProfiling;
  ur_exp_command_buffer_handle_t OutCommandBuffer;
  ur_exp_command_buffer_desc_t Desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                    nullptr, MIsUpdatable,
                                    IsInOrderCommandBuffer, MEnableProfiling};
  context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(MContext);
  sycl::detail::adapter_impl &Adapter = ContextImpl.getAdapter();
  sycl::detail::device_impl &DeviceImpl = *sycl::detail::getSyclObjImpl(Device);
  ur_result_t Res =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urCommandBufferCreateExp>(
          ContextImpl.getHandleRef(), DeviceImpl.getHandleRef(), &Desc,
          &OutCommandBuffer);
  if (Res != UR_RESULT_SUCCESS) {
    throw sycl::exception(errc::invalid, "Failed to create UR command-buffer");
  }

  Partition->MCommandBuffers[Device] = OutCommandBuffer;

  for (node_impl &Node : Partition->schedule()) {
    // Some nodes are not scheduled like other nodes, and only their
    // dependencies are propagated in findRealDeps
    if (!Node.requiresEnqueue())
      continue;

    sycl::detail::CGType type = Node.MCGType;
    // If the node is a kernel with no special requirements we can enqueue it
    // directly.
    if (type == sycl::detail::CGType::Kernel &&
        Node.MCommandGroup->getRequirements().size() +
                static_cast<sycl::detail::CGExecKernel *>(
                    Node.MCommandGroup.get())
                    ->MStreams.size() ==
            0) {
      if (auto OptSyncPoint =
              enqueueNodeDirect(MContext, DeviceImpl, OutCommandBuffer, Node,
                                IsInOrderCommandBuffer)) {
        assert(!IsInOrderCommandBuffer &&
               "In-order partitions should not create a sync point");
        MSyncPoints[&Node] = *OptSyncPoint;
      }
    } else {
      if (auto OptSyncPoint =
              enqueueNode(OutCommandBuffer, Node, IsInOrderCommandBuffer)) {
        assert(!IsInOrderCommandBuffer &&
               "In-order partitions should not create a sync point");
        MSyncPoints[&Node] = *OptSyncPoint;
      }
    }
  }

  Res =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urCommandBufferFinalizeExp>(
          OutCommandBuffer);
  if (Res != UR_RESULT_SUCCESS) {
    throw sycl::exception(errc::invalid,
                          "Failed to finalize UR command-buffer");
  }
}

exec_graph_impl::exec_graph_impl(sycl::context Context,
                                 const std::shared_ptr<graph_impl> &GraphImpl,
                                 const property_list &PropList)
    : MSchedule(), MGraphImpl(GraphImpl), MSyncPoints(),
      MDevice(GraphImpl->getDevice()), MContext(Context), MRequirements(),
      MSchedulerDependencies(),
      MIsUpdatable(PropList.has_property<property::graph::updatable>()),
      MEnableProfiling(
          PropList.has_property<property::graph::enable_profiling>()),
      MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {
  checkGraphPropertiesAndThrow(PropList);
  // If the graph has been marked as updatable then check if the backend
  // actually supports that. Devices supporting aspect::ext_oneapi_graph must
  // have support for graph update.
  if (MIsUpdatable) {
    bool SupportsUpdate = MGraphImpl->getDevice().has(aspect::ext_oneapi_graph);
    if (!SupportsUpdate) {
      throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                            "Device does not support Command Graph update");
    }
  }
  // Copy nodes from GraphImpl and merge any subgraph nodes into this graph.
  duplicateNodes();

  if (auto PlaceholderQueuePtr = GraphImpl->getQueue()) {
    MQueueImpl = std::move(PlaceholderQueuePtr);
  } else {
    MQueueImpl = sycl::detail::queue_impl::create(
        *sycl::detail::getSyclObjImpl(GraphImpl->getDevice()),
        *sycl::detail::getSyclObjImpl(Context), sycl::async_handler{},
        sycl::property_list{});
  }
}

exec_graph_impl::~exec_graph_impl() {
  try {
    MGraphImpl->markExecGraphDestroyed();

    sycl::detail::adapter_impl &Adapter =
        sycl::detail::getSyclObjImpl(MContext)->getAdapter();
    MSchedule.clear();

    // Clean up any graph-owned allocations that were allocated
    MGraphImpl->getMemPool().deallocateAndUnmapAll();

    for (const auto &Partition : MPartitions) {
      Partition->MSchedule.clear();
      for (const auto &Iter : Partition->MCommandBuffers) {
        if (auto CmdBuf = Iter.second; CmdBuf) {
          ur_result_t Res = Adapter.call_nocheck<
              sycl::detail::UrApiKind::urCommandBufferReleaseExp>(CmdBuf);
          (void)Res;
          assert(Res == UR_RESULT_SUCCESS);
        }
      }
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~exec_graph_impl", e);
  }
}

// Clean up any execution events which have finished so we don't pass them
// to the scheduler.
static void cleanupExecutionEvents(std::vector<EventImplPtr> &ExecutionEvents) {

  auto Predicate = [](EventImplPtr &EventPtr) {
    return EventPtr->isCompleted();
  };

  ExecutionEvents.erase(
      std::remove_if(ExecutionEvents.begin(), ExecutionEvents.end(), Predicate),
      ExecutionEvents.end());
}

EventImplPtr exec_graph_impl::enqueueHostTaskPartition(
    std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
    sycl::detail::CG::StorageInitHelper CGData, bool EventNeeded) {

  auto NodeImpl = Partition->MSchedule.front();
  auto NodeCommandGroup =
      static_cast<sycl::detail::CGHostTask *>(NodeImpl->MCommandGroup.get());

  CGData.MRequirements.insert(CGData.MRequirements.end(),
                              NodeCommandGroup->getRequirements().begin(),
                              NodeCommandGroup->getRequirements().end());
  CGData.MAccStorage.insert(CGData.MAccStorage.end(),
                            NodeCommandGroup->getAccStorage().begin(),
                            NodeCommandGroup->getAccStorage().end());

  assert(std::all_of(
      NodeCommandGroup->MArgs.begin(), NodeCommandGroup->MArgs.end(),
      [](ArgDesc Arg) {
        return Arg.MType != sycl::detail::kernel_param_kind_t::kind_std_layout;
      }));

  // Create a copy of this node command-group which contains the right
  // dependencies for the current execution.
  std::unique_ptr<sycl::detail::CG> CommandGroup =
      std::make_unique<sycl::detail::CGHostTask>(sycl::detail::CGHostTask(
          NodeCommandGroup->MHostTask, &Queue, NodeCommandGroup->MContext.get(),
          NodeCommandGroup->MArgs, std::move(CGData),
          NodeCommandGroup->getType()));

  EventImplPtr SchedulerEvent = sycl::detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), Queue, EventNeeded);

  if (EventNeeded) {
    return SchedulerEvent;
  }
  return nullptr;
}

EventImplPtr exec_graph_impl::enqueuePartitionWithScheduler(
    std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
    sycl::detail::CG::StorageInitHelper CGData, bool EventNeeded) {

  if (!Partition->MRequirements.empty()) {
    CGData.MRequirements.insert(CGData.MRequirements.end(),
                                Partition->MRequirements.begin(),
                                Partition->MRequirements.end());
    CGData.MAccStorage.insert(CGData.MAccStorage.end(),
                              Partition->MAccessors.begin(),
                              Partition->MAccessors.end());
  }

  auto CommandBuffer = Partition->MCommandBuffers[Queue.get_device()];

  std::unique_ptr<sycl::detail::CG> CommandGroup =
      std::make_unique<sycl::detail::CGExecCommandBuffer>(
          CommandBuffer, nullptr, std::move(CGData));

  EventImplPtr SchedulerEvent = sycl::detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), Queue, EventNeeded);

  if (EventNeeded) {
    SchedulerEvent->setEventFromSubmittedExecCommandBuffer(true);
    return SchedulerEvent;
  }

  return nullptr;
}

EventImplPtr exec_graph_impl::enqueuePartitionDirectly(
    std::shared_ptr<partition> &Partition, sycl::detail::queue_impl &Queue,
    std::vector<detail::EventImplPtr> &WaitEvents, bool EventNeeded) {

  // Create a list containing all the UR event handles in WaitEvents. WaitEvents
  // is assumed to be safe for scheduler bypass and any host-task events that it
  // contains can be ignored.
  std::vector<ur_event_handle_t> UrEventHandles{};
  UrEventHandles.reserve(WaitEvents.size());
  for (auto &SyclWaitEvent : WaitEvents) {
    if (auto URHandle = SyclWaitEvent->getHandle()) {
      UrEventHandles.push_back(URHandle);
    }
  }

  auto CommandBuffer = Partition->MCommandBuffers[Queue.get_device()];
  const size_t UrEnqueueWaitListSize = UrEventHandles.size();
  const ur_event_handle_t *UrEnqueueWaitList =
      UrEnqueueWaitListSize == 0 ? nullptr : UrEventHandles.data();

  if (!EventNeeded) {
    Queue.getAdapter().call<sycl::detail::UrApiKind::urEnqueueCommandBufferExp>(
        Queue.getHandleRef(), CommandBuffer, UrEnqueueWaitListSize,
        UrEnqueueWaitList, nullptr);
    return nullptr;
  } else {
    auto NewEvent = sycl::detail::event_impl::create_device_event(Queue);
    NewEvent->setContextImpl(Queue.getContextImpl());
    NewEvent->setStateIncomplete();
    NewEvent->setSubmissionTime();
    ur_event_handle_t UrEvent = nullptr;
    Queue.getAdapter().call<sycl::detail::UrApiKind::urEnqueueCommandBufferExp>(
        Queue.getHandleRef(), CommandBuffer, UrEventHandles.size(),
        UrEnqueueWaitList, &UrEvent);
    NewEvent->setHandle(UrEvent);
    NewEvent->setEventFromSubmittedExecCommandBuffer(true);
    return NewEvent;
  }
}

EventImplPtr
exec_graph_impl::enqueuePartitions(sycl::detail::queue_impl &Queue,
                                   sycl::detail::CG::StorageInitHelper &CGData,
                                   bool IsCGDataSafeForSchedulerBypass,
                                   bool EventNeeded) {

  // If EventNeeded is true, this vector is used to keep track of dependencies
  // for the returned event. This is used when the graph has multiple end nodes
  // which cannot be tracked with a single scheduler event.
  std::vector<EventImplPtr> PostCompleteDependencies;
  // TODO After refactoring the event class to use enable_shared_from_this, the
  // events used in PostCompleteDependencies can become raw pointers as long as
  // Event->attachEventToComplete() extends the lifetime of the pointer with
  // shared_from_this.

  // This variable represents the returned event. It will always be nullptr if
  // EventNeeded is false.
  EventImplPtr SignalEvent;

  // CGData.MEvents gets cleared after every partition enqueue. If we need the
  // original events, a backup needs to be created now. This is only needed when
  // the graph contains more than one root partition.
  std::vector<detail::EventImplPtr> BackupCGDataEvents;
  if (MRootPartitions.size() > 1) {
    BackupCGDataEvents = CGData.MEvents;
  }

  for (auto &Partition : MPartitions) {

    if (Partition->MPredecessors.empty() && CGData.MEvents.empty()) {
      // If this is a root partition and CGData has been cleared already, we
      // need to restore it so that the partition execution waits for the
      // dependencies of this graph execution.
      CGData.MEvents = BackupCGDataEvents;
    } else {
      // Partitions can have multiple dependencies from previously executed
      // partitions. To enforce this ordering, we need to add these dependencies
      // to CGData.
      for (auto &Predecessor : Partition->MPredecessors) {
        CGData.MEvents.push_back(Predecessor->MEvent);
      }
    }

    // We always need to request an event to use as dependency between
    // partitions executions and between graph executions because the
    // scheduler doesn't seem to guarantee the execution order of host-tasks
    // without adding explicit event dependencies even when the queue is
    // in-order.
    constexpr bool RequestEvent = true;

    EventImplPtr EnqueueEvent;
    if (Partition->MIsHostTask) {
      // The event returned by a host-task is always needed to synchronize with
      // other partitions or to be used by the sycl queue as a dependency for
      // further commands.
      EnqueueEvent =
          enqueueHostTaskPartition(Partition, Queue, CGData, RequestEvent);
    } else {
      // The scheduler can only be skipped if the partition is a root and is not
      // a host-task. This is because all host-tasks need to go through the
      // scheduler and, since only the scheduler can wait on host-task events,
      // any subsequent partitions that depend on a host-task partition also
      // need to use the scheduler.
      bool SkipScheduler = Partition->MPredecessors.empty() &&
                           IsCGDataSafeForSchedulerBypass &&
                           Partition->MRequirements.empty();
      if (SkipScheduler) {
        EnqueueEvent = enqueuePartitionDirectly(Partition, Queue,
                                                CGData.MEvents, RequestEvent);
      } else {
        EnqueueEvent = enqueuePartitionWithScheduler(Partition, Queue, CGData,
                                                     RequestEvent);
      }
    }

    if (!Partition->MSuccessors.empty()) {
      // Need to keep track of the EnqueueEvent for this partition so that
      // it can be added as a dependency to CGData when successors are executed.
      Partition->MEvent = std::move(EnqueueEvent);
    } else {
      // Unified runtime guarantees the execution order of command-buffers.
      // However, since host-tasks have been scheduled, we always need to add a
      // dependency for the next graph execution. If we don't the next graph
      // execution could end up with the same host-task node executing in
      // parallel.
      MSchedulerDependencies.push_back(EnqueueEvent);
      if (EventNeeded) {
        const bool IsLastPartition = (Partition == MPartitions.back());
        if (IsLastPartition) {
          // If we are in the last partition move the event to SignalEvent,
          // so that it can be returned to the user.
          SignalEvent = std::move(EnqueueEvent);
        } else {
          // If it's not the last partition, keep track of the event as a post
          // complete dependency.
          PostCompleteDependencies.push_back(std::move(EnqueueEvent));
        }
      }
    }

    // Clear the event list so that unnecessary dependencies are not added on
    // future partition executions.
    CGData.MEvents.clear();
  }

  if (EventNeeded) {
    for (auto &EventFromOtherPartitions : PostCompleteDependencies) {
      SignalEvent->attachEventToComplete(EventFromOtherPartitions);
    }
  }

  return SignalEvent;
}

std::pair<EventImplPtr, bool>
exec_graph_impl::enqueue(sycl::detail::queue_impl &Queue,
                         sycl::detail::CG::StorageInitHelper CGData,
                         bool EventNeeded) {
  WriteLock Lock(MMutex);

  cleanupExecutionEvents(MSchedulerDependencies);
  CGData.MEvents.insert(CGData.MEvents.end(), MSchedulerDependencies.begin(),
                        MSchedulerDependencies.end());
  bool IsCGDataSafeForSchedulerBypass =
      detail::Scheduler::areEventsSafeForSchedulerBypass(
          CGData.MEvents, Queue.getContextImpl()) &&
      CGData.MRequirements.empty();
  bool SkipScheduler = IsCGDataSafeForSchedulerBypass && !MContainsHostTask;

  // This variable represents the returned event. It will always be nullptr if
  // EventNeeded is false.
  EventImplPtr SignalEvent;
  if (!MContainsHostTask) {
    SkipScheduler = SkipScheduler && MPartitions[0]->MRequirements.empty();
    if (SkipScheduler) {
      SignalEvent = enqueuePartitionDirectly(MPartitions[0], Queue,
                                             CGData.MEvents, EventNeeded);
    } else {
      bool RequestSchedulerEvent = EventNeeded || MIsUpdatable;
      auto SchedulerEvent = enqueuePartitionWithScheduler(
          MPartitions[0], Queue, std::move(CGData), RequestSchedulerEvent);

      // If the graph is updatable, and we are going through the scheduler, we
      // need to track the execution event to make sure that any future updates
      // happen after the graph execution.
      // There is no need to track the execution event when updates are not
      // allowed because Unified Runtime already guarantees the execution order
      // of command-buffers.
      if (MIsUpdatable) {
        MSchedulerDependencies.push_back(
            EventNeeded ? SchedulerEvent : std::move(SchedulerEvent));
      }

      if (EventNeeded) {
        SignalEvent = std::move(SchedulerEvent);
      }
    }
  } else {
    SignalEvent = enqueuePartitions(
        Queue, CGData, IsCGDataSafeForSchedulerBypass, EventNeeded);
  }

  if (EventNeeded) {
    SignalEvent->setProfilingEnabled(MEnableProfiling);
  }

  return {SignalEvent, SkipScheduler};
}

void exec_graph_impl::duplicateNodes() {
  // Map of original modifiable nodes (keys) to new duplicated nodes (values)
  std::unordered_map<node_impl *, node_impl *> NodesMap;
  nodes_range ModifiableNodes{MGraphImpl->MNodeStorage};
  std::vector<std::shared_ptr<node_impl>> NewNodes;

  const size_t NodeCount = ModifiableNodes.size();
  NodesMap.reserve(NodeCount);
  NewNodes.reserve(NodeCount);

  bool foundSubgraph = false;

  for (node_impl &OriginalNode : ModifiableNodes) {
    NewNodes.push_back(std::make_shared<node_impl>(OriginalNode));
    node_impl &NodeCopy = *NewNodes.back();

    foundSubgraph |= (NodeCopy.MNodeType == node_type::subgraph);

    // Associate the ID of the original node with the node copy for later quick
    // access
    MIDCache.insert(std::make_pair(OriginalNode.MID, &NodeCopy));

    // Clear edges between nodes so that we can replace with new ones
    NodeCopy.MSuccessors.clear();
    NodeCopy.MPredecessors.clear();
    // Push the new node to the front of the stack
    // Associate the new node with the old one for updating edges
    NodesMap.insert({&OriginalNode, &NodeCopy});
  }

  // Now that all nodes have been copied rebuild edges on new nodes. This must
  // be done as a separate step since successors may be out of order.
  auto OrigIt = ModifiableNodes.begin(), OrigEnd = ModifiableNodes.end();
  for (auto NewIt = NewNodes.begin(); OrigIt != OrigEnd; ++OrigIt, ++NewIt) {
    node_impl &OriginalNode = *OrigIt;
    node_impl &NodeCopy = **NewIt;
    // Look through all the original node successors, find their copies and
    // register those as successors with the current copied node
    for (node_impl &NextNode : OriginalNode.successors()) {
      node_impl &Successor = *NodesMap.at(&NextNode);
      NodeCopy.registerSuccessor(Successor);
    }
  }

  // Subgraph nodes need special handling, we extract all subgraph nodes and
  // merge them into the main node list
  if (foundSubgraph) {
    for (auto NewNodeIt = NewNodes.rbegin(); NewNodeIt != NewNodes.rend();
         ++NewNodeIt) {
      auto NewNode = *NewNodeIt;
      if (NewNode->MNodeType != node_type::subgraph) {
        continue;
      }
      nodes_range SubgraphNodes{NewNode->MSubGraphImpl->MNodeStorage};
      std::deque<std::shared_ptr<node_impl>> NewSubgraphNodes{};

      // Map of original subgraph nodes (keys) to new duplicated nodes (values)
      std::map<node_impl *, node_impl *> SubgraphNodesMap;

      // Copy subgraph nodes
      for (node_impl &SubgraphNode : SubgraphNodes) {
        NewSubgraphNodes.push_back(std::make_shared<node_impl>(SubgraphNode));
        node_impl &NodeCopy = *NewSubgraphNodes.back();
        // Associate the ID of the original subgraph node with all extracted
        // node copies for future quick access.
        MIDCache.insert(std::make_pair(SubgraphNode.MID, &NodeCopy));

        SubgraphNodesMap.insert({&SubgraphNode, &NodeCopy});
        NodeCopy.MSuccessors.clear();
        NodeCopy.MPredecessors.clear();
      }

      // Rebuild edges for new subgraph nodes
      auto OrigIt = SubgraphNodes.begin(), OrigEnd = SubgraphNodes.end();
      for (auto NewIt = NewSubgraphNodes.begin(); OrigIt != OrigEnd;
           ++OrigIt, ++NewIt) {
        node_impl &SubgraphNode = *OrigIt;
        node_impl &NodeCopy = **NewIt;

        for (node_impl &NextNode : SubgraphNode.successors()) {
          node_impl &Successor = *SubgraphNodesMap.at(&NextNode);
          NodeCopy.registerSuccessor(Successor);
        }
      }

      // Collect input and output nodes for the subgraph
      std::vector<node_impl *> Inputs;
      std::vector<node_impl *> Outputs;
      for (std::shared_ptr<node_impl> &NodeImpl : NewSubgraphNodes) {
        if (NodeImpl->MPredecessors.size() == 0) {
          Inputs.push_back(&*NodeImpl);
        }
        if (NodeImpl->MSuccessors.size() == 0) {
          Outputs.push_back(&*NodeImpl);
        }
      }

      // Update the predecessors and successors of the nodes which reference the
      // original subgraph node

      // Predecessors
      for (node_impl &PredNode : NewNode->predecessors()) {
        auto &Successors = PredNode.MSuccessors;

        // Remove the subgraph node from this nodes successors
        Successors.erase(
            std::remove(Successors.begin(), Successors.end(), NewNode.get()),
            Successors.end());

        // Add all input nodes from the subgraph as successors for this node
        // instead
        for (node_impl *Input : Inputs) {
          PredNode.registerSuccessor(*Input);
        }
      }

      // Successors
      for (node_impl &SuccNode : NewNode->successors()) {
        auto &Predecessors = SuccNode.MPredecessors;

        // Remove the subgraph node from this nodes successors
        Predecessors.erase(std::remove(Predecessors.begin(), Predecessors.end(),
                                       NewNode.get()),
                           Predecessors.end());

        // Add all Output nodes from the subgraph as predecessors for this node
        // instead
        for (node_impl *Output : Outputs) {
          Output->registerSuccessor(SuccNode);
        }
      }

      // Remove single subgraph node and add all new individual subgraph nodes
      // to the node storage in its place
      auto OldPositionIt =
          NewNodes.erase(std::find(NewNodes.begin(), NewNodes.end(), NewNode));
      // Also set the iterator to the newly added nodes so we can continue
      // iterating over all remaining nodes
      auto InsertIt = NewNodes.insert(
          OldPositionIt, std::make_move_iterator(NewSubgraphNodes.begin()),
          std::make_move_iterator(NewSubgraphNodes.end()));
      // Since the new reverse_iterator will be at i - 1 we need to advance it
      // when constructing
      NewNodeIt = std::make_reverse_iterator(std::next(InsertIt));
    }
  }

  // Store all the new nodes locally
  MNodeStorage = std::move(NewNodes);
}

void exec_graph_impl::update(std::shared_ptr<graph_impl> GraphImpl) {

  if (MDevice != GraphImpl->getDevice()) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "Cannot update using a graph created with a different device.");
  }
  if (MContext != GraphImpl->getContext()) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "Cannot update using a graph created with a different context.");
  }

  if (MNodeStorage.size() != GraphImpl->MNodeStorage.size()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Cannot update using a graph with a different "
                          "topology. Mismatch found in the number of nodes.");
  }

  for (uint32_t i = 0; i < MNodeStorage.size(); ++i) {
    if (MNodeStorage[i]->MSuccessors.size() !=
            GraphImpl->MNodeStorage[i]->MSuccessors.size() ||
        MNodeStorage[i]->MPredecessors.size() !=
            GraphImpl->MNodeStorage[i]->MPredecessors.size()) {
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "Cannot update using a graph with a different topology. Mismatch "
          "found in the number of edges.");
    }
    if (MNodeStorage[i]->MCGType != GraphImpl->MNodeStorage[i]->MCGType) {
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "Cannot update using a graph with mismatched node types. Each pair "
          "of nodes being updated must have the same type");
    }

    if (MNodeStorage[i]->MCGType == sycl::detail::CGType::Kernel) {
      sycl::detail::CGExecKernel *TargetCGExec =
          static_cast<sycl::detail::CGExecKernel *>(
              MNodeStorage[i]->MCommandGroup.get());
      std::string_view TargetKernelName = TargetCGExec->getKernelName();

      sycl::detail::CGExecKernel *SourceCGExec =
          static_cast<sycl::detail::CGExecKernel *>(
              GraphImpl->MNodeStorage[i]->MCommandGroup.get());
      std::string_view SourceKernelName = SourceCGExec->getKernelName();

      if (TargetKernelName != SourceKernelName) {
        std::stringstream ErrorStream(
            "Cannot update using a graph with mismatched kernel "
            "types. Source node type ");
        ErrorStream << SourceKernelName;
        ErrorStream << ", target node type ";
        ErrorStream << TargetKernelName;
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              ErrorStream.str());
      }
    }
  }

  for (uint32_t i = 0; i < MNodeStorage.size(); ++i) {
    MIDCache.insert(
        std::make_pair(GraphImpl->MNodeStorage[i]->MID, MNodeStorage[i].get()));
  }

  update(GraphImpl->nodes());
}

void exec_graph_impl::update(node_impl &Node) {
  this->update(std::vector<node_impl *>{&Node});
}

void exec_graph_impl::update(nodes_range Nodes) {
  if (!MIsUpdatable) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "update() cannot be called on a executable graph "
                          "which was not created with property::updatable");
  }

  // If the graph contains host tasks we need special handling here because
  // their state lives in the graph object itself, so we must do the update
  // immediately here. Whereas all other command state lives in the backend so
  // it can be scheduled along with other commands.
  if (MContainsHostTask) {
    updateHostTasksImpl(Nodes);
  }

  // If there are any accessor requirements, we have to update through the
  // scheduler to ensure that any allocations have taken place before trying
  // to update.
  std::vector<sycl::detail::AccessorImplHost *> UpdateRequirements;
  bool NeedScheduledUpdate = needsScheduledUpdate(Nodes, UpdateRequirements);
  if (NeedScheduledUpdate) {
    cleanupExecutionEvents(MSchedulerDependencies);

    // Track the event for the update command since execution may be blocked by
    // other scheduler commands
    auto UpdateEvent =
        sycl::detail::Scheduler::getInstance().addCommandGraphUpdate(
            this, Nodes, MQueueImpl.get(), std::move(UpdateRequirements),
            MSchedulerDependencies);

    MSchedulerDependencies.push_back(UpdateEvent);

    if (MContainsHostTask) {
      // If the graph has HostTasks, the update has to be blocking. This is
      // needed because HostTask nodes (and all the nodes that depend on
      // HostTasks), are scheduled using a separate thread. This wait call
      // acts as a synchronization point for that thread.
      UpdateEvent->wait();
    }
  } else {
    // For each partition in the executable graph, call UR update on the
    // command-buffer with the nodes to update.
    auto PartitionedNodes = getURUpdatableNodes(Nodes);
    for (auto &[PartitionIndex, NodeImpl] : PartitionedNodes) {
      auto &Partition = MPartitions[PartitionIndex];
      auto CommandBuffer = Partition->MCommandBuffers[MDevice];
      updateURImpl(CommandBuffer, NodeImpl);
    }
  }

  // Rebuild cached requirements and accessor storage for this graph with
  // updated nodes
  MRequirements.clear();
  for (auto &Partition : MPartitions) {
    Partition->MRequirements.clear();
    Partition->MAccessors.clear();
  }
  buildRequirements();
}

bool exec_graph_impl::needsScheduledUpdate(
    nodes_range Nodes,
    std::vector<sycl::detail::AccessorImplHost *> &UpdateRequirements) {
  // If there are any accessor requirements, we have to update through the
  // scheduler to ensure that any allocations have taken place before trying to
  // update.
  bool NeedScheduledUpdate = false;
  // At worst we may have as many requirements as there are for the entire graph
  // for updating.
  UpdateRequirements.reserve(MRequirements.size());
  for (node_impl &Node : Nodes) {
    // Check if node(s) derived from this modifiable node exists in this graph
    if (MIDCache.count(Node.getID()) == 0) {
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "Node passed to update() is not part of the graph.");
    }

    if (!Node.isUpdatable()) {
      std::string ErrorString = "node_type::";
      ErrorString += nodeTypeToString(Node.MNodeType);
      ErrorString +=
          " nodes are not supported for update. Only kernel, host_task, "
          "barrier and empty nodes are supported.";
      throw sycl::exception(errc::invalid, ErrorString);
    }

    if (const auto &CG = Node.MCommandGroup;
        CG && CG->getRequirements().size() != 0) {
      NeedScheduledUpdate = true;

      UpdateRequirements.insert(UpdateRequirements.end(),
                                Node.MCommandGroup->getRequirements().begin(),
                                Node.MCommandGroup->getRequirements().end());
    }
  }

  // If we have previous execution events do the update through the scheduler to
  // ensure it is ordered correctly.
  NeedScheduledUpdate |= MSchedulerDependencies.size() > 0;

  return NeedScheduledUpdate;
}

void exec_graph_impl::populateURKernelUpdateStructs(
    node_impl &Node, FastKernelCacheValPtr &BundleObjs,
    std::vector<ur_exp_command_buffer_update_memobj_arg_desc_t> &MemobjDescs,
    std::vector<ur_kernel_arg_mem_obj_properties_t> &MemobjProps,
    std::vector<ur_exp_command_buffer_update_pointer_arg_desc_t> &PtrDescs,
    std::vector<ur_exp_command_buffer_update_value_arg_desc_t> &ValueDescs,
    sycl::detail::NDRDescT &NDRDesc,
    ur_exp_command_buffer_update_kernel_launch_desc_t &UpdateDesc) const {
  sycl::detail::context_impl &ContextImpl =
      *sycl::detail::getSyclObjImpl(MContext);
  sycl::detail::adapter_impl &Adapter = ContextImpl.getAdapter();
  sycl::detail::device_impl &DeviceImpl =
      *sycl::detail::getSyclObjImpl(MGraphImpl->getDevice());

  // Gather arg information from Node
  auto &ExecCG =
      *(static_cast<sycl::detail::CGExecKernel *>(Node.MCommandGroup.get()));
  // Copy args because we may modify them
  std::vector<sycl::detail::ArgDesc> NodeArgs = ExecCG.getArguments();
  // Copy NDR desc since we need to modify it
  NDRDesc = ExecCG.MNDRDesc;

  ur_kernel_handle_t UrKernel = nullptr;
  auto Kernel = ExecCG.MSyclKernel;
  auto KernelBundleImplPtr = ExecCG.MKernelBundle;
  const sycl::detail::KernelArgMask *EliminatedArgMask = nullptr;

  if (Kernel != nullptr) {
    UrKernel = Kernel->getHandleRef();
    EliminatedArgMask = Kernel->getKernelArgMask();
  } else if (auto SyclKernelImpl =
                 KernelBundleImplPtr ? KernelBundleImplPtr->tryGetKernel(
                                           ExecCG.MDeviceKernelInfo.Name)
                                     : std::shared_ptr<kernel_impl>{nullptr}) {
    UrKernel = SyclKernelImpl->getHandleRef();
    EliminatedArgMask = SyclKernelImpl->getKernelArgMask();
  } else {
    BundleObjs = sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
        ContextImpl, DeviceImpl, ExecCG.MDeviceKernelInfo);
    UrKernel = BundleObjs->MKernelHandle;
    EliminatedArgMask = BundleObjs->MKernelArgMask;
  }

  // Remove eliminated args
  std::vector<sycl::detail::ArgDesc> MaskedArgs;
  MaskedArgs.reserve(NodeArgs.size());

  sycl::detail::applyFuncOnFilteredArgs(
      EliminatedArgMask, NodeArgs,
      [&MaskedArgs](sycl::detail::ArgDesc &Arg, int NextTrueIndex) {
        MaskedArgs.emplace_back(Arg.MType, Arg.MPtr, Arg.MSize, NextTrueIndex);
      });

  // Reverse kernel dims
  sycl::detail::ReverseRangeDimensionsForKernel(NDRDesc);

  size_t RequiredWGSize[3] = {0, 0, 0};
  size_t *LocalSize = nullptr;

  if (NDRDesc.LocalSize[0] != 0)
    LocalSize = &NDRDesc.LocalSize[0];
  else {
    Adapter.call<sycl::detail::UrApiKind::urKernelGetGroupInfo>(
        UrKernel, DeviceImpl.getHandleRef(),
        UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize,
        /* param_value_size_ret = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }

  // Storage for individual arg descriptors
  MemobjDescs.reserve(MaskedArgs.size());
  PtrDescs.reserve(MaskedArgs.size());
  ValueDescs.reserve(MaskedArgs.size());
  MemobjProps.resize(MaskedArgs.size()); // resize since we access by reference

  UpdateDesc.stype =
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC;
  UpdateDesc.pNext = nullptr;

  // Collect arg descriptors and fill kernel launch descriptor
  using sycl::detail::kernel_param_kind_t;
  for (size_t i = 0; i < MaskedArgs.size(); i++) {
    auto &NodeArg = MaskedArgs[i];
    switch (NodeArg.MType) {
    case kernel_param_kind_t::kind_pointer: {
      PtrDescs.push_back(
          {UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC,
           nullptr, static_cast<uint32_t>(NodeArg.MIndex), nullptr,
           NodeArg.MPtr});
    } break;
    case kernel_param_kind_t::kind_std_layout: {
      ValueDescs.push_back(
          {UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, nullptr,
           static_cast<uint32_t>(NodeArg.MIndex),
           static_cast<uint32_t>(NodeArg.MSize), nullptr, NodeArg.MPtr});
    } break;
    case kernel_param_kind_t::kind_accessor: {
      sycl::detail::Requirement *Req =
          static_cast<sycl::detail::Requirement *>(NodeArg.MPtr);

      ur_kernel_arg_mem_obj_properties_t &MemObjProp = MemobjProps[i];
      MemObjProp.stype = UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES;
      MemObjProp.pNext = nullptr;
      switch (Req->MAccessMode) {
      case access::mode::read: {
        MemObjProp.memoryAccess = UR_MEM_FLAG_READ_ONLY;
        break;
      }
      case access::mode::write:
      case access::mode::discard_write: {
        MemObjProp.memoryAccess = UR_MEM_FLAG_WRITE_ONLY;
        break;
      }
      default: {
        MemObjProp.memoryAccess = UR_MEM_FLAG_READ_WRITE;
        break;
      }
      }
      MemobjDescs.push_back(
          {UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, nullptr,
           static_cast<uint32_t>(NodeArg.MIndex), &MemObjProp,
           static_cast<ur_mem_handle_t>(Req->MData)});

    } break;

    default:
      break;
    }
  }

  UpdateDesc.hNewKernel = UrKernel;
  UpdateDesc.numNewMemObjArgs = MemobjDescs.size();
  UpdateDesc.pNewMemObjArgList = MemobjDescs.data();
  UpdateDesc.numNewPointerArgs = PtrDescs.size();
  UpdateDesc.pNewPointerArgList = PtrDescs.data();
  UpdateDesc.numNewValueArgs = ValueDescs.size();
  UpdateDesc.pNewValueArgList = ValueDescs.data();

  UpdateDesc.pNewGlobalWorkOffset = &NDRDesc.GlobalOffset[0];
  UpdateDesc.pNewGlobalWorkSize = &NDRDesc.GlobalSize[0];
  UpdateDesc.pNewLocalWorkSize = LocalSize;
  UpdateDesc.newWorkDim = NDRDesc.Dims;

  // Query the ID cache to find the equivalent exec node for the node passed to
  // this function.
  // TODO: Handle subgraphs or any other cases where multiple nodes may be
  // associated with a single key, once those node types are supported for
  // update.
  auto ExecNode = MIDCache.find(Node.MID);
  assert(ExecNode != MIDCache.end() && "Node ID was not found in ID cache");

  auto Command = MCommandMap.find(ExecNode->second);
  assert(Command != MCommandMap.end());
  UpdateDesc.hCommand = Command->second;

  // Update ExecNode with new values from Node, in case we ever need to
  // rebuild the command buffers
  ExecNode->second->updateFromOtherNode(Node);
}

std::map<int, std::vector<node_impl *>>
exec_graph_impl::getURUpdatableNodes(nodes_range Nodes) const {
  // Iterate over the list of nodes, and for every node that can
  // be updated through UR, add it to the list of nodes for
  // that can be updated for the UR command-buffer partition.
  std::map<int, std::vector<node_impl *>> PartitionedNodes;

  // Initialize vector for each partition
  for (size_t i = 0; i < MPartitions.size(); i++) {
    PartitionedNodes[i] = {};
  }

  for (node_impl &Node : Nodes) {
    // Kernel node update is the only command type supported in UR for update.
    if (Node.MCGType != sycl::detail::CGType::Kernel) {
      continue;
    }

    auto ExecNode = MIDCache.find(Node.MID);
    assert(ExecNode != MIDCache.end() && "Node ID was not found in ID cache");
    auto PartitionIndex = MPartitionNodes.find(ExecNode->second);
    assert(PartitionIndex != MPartitionNodes.end());
    PartitionedNodes[PartitionIndex->second].push_back(&Node);
  }

  return PartitionedNodes;
}

void exec_graph_impl::updateHostTasksImpl(nodes_range Nodes) const {
  for (node_impl &Node : Nodes) {
    if (Node.MNodeType != node_type::host_task) {
      continue;
    }
    // Query the ID cache to find the equivalent exec node for the node passed
    // to this function.
    auto ExecNode = MIDCache.find(Node.MID);
    assert(ExecNode != MIDCache.end() && "Node ID was not found in ID cache");

    ExecNode->second->updateFromOtherNode(Node);
  }
}

void exec_graph_impl::updateURImpl(ur_exp_command_buffer_handle_t CommandBuffer,
                                   nodes_range Nodes) const {
  const size_t NumUpdatableNodes = Nodes.size();
  if (NumUpdatableNodes == 0) {
    return;
  }

  // The urCommandBufferUpdateKernelLaunchExp API takes structs which contain
  // members that are pointers to other structs. The lifetime of all the
  // pointers (including nested pointers) needs to be valid at the time of the
  // urCommandBufferUpdateKernelLaunchExp call. Define the objects here which
  // will be populated and used in the urCommandBufferUpdateKernelLaunchExp
  // call.
  std::vector<std::vector<ur_exp_command_buffer_update_memobj_arg_desc_t>>
      MemobjDescsList(NumUpdatableNodes);
  std::vector<std::vector<ur_kernel_arg_mem_obj_properties_t>> MemobjPropsList(
      NumUpdatableNodes);
  std::vector<std::vector<ur_exp_command_buffer_update_pointer_arg_desc_t>>
      PtrDescsList(NumUpdatableNodes);
  std::vector<std::vector<ur_exp_command_buffer_update_value_arg_desc_t>>
      ValueDescsList(NumUpdatableNodes);
  std::vector<sycl::detail::NDRDescT> NDRDescList(NumUpdatableNodes);
  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> UpdateDescList(
      NumUpdatableNodes);
  std::vector<FastKernelCacheValPtr> KernelBundleObjList(NumUpdatableNodes);

  size_t StructListIndex = 0;
  for (node_impl &Node : Nodes) {
    // This should be the case when getURUpdatableNodes() is used to
    // create the list of nodes.
    assert(Node.MCGType == sycl::detail::CGType::Kernel);

    auto &MemobjDescs = MemobjDescsList[StructListIndex];
    auto &MemobjProps = MemobjPropsList[StructListIndex];
    auto &KernelBundleObjs = KernelBundleObjList[StructListIndex];
    auto &PtrDescs = PtrDescsList[StructListIndex];
    auto &ValueDescs = ValueDescsList[StructListIndex];
    auto &NDRDesc = NDRDescList[StructListIndex];
    auto &UpdateDesc = UpdateDescList[StructListIndex];
    populateURKernelUpdateStructs(Node, KernelBundleObjs, MemobjDescs,
                                  MemobjProps, PtrDescs, ValueDescs, NDRDesc,
                                  UpdateDesc);
    StructListIndex++;
  }

  context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(MContext);
  sycl::detail::adapter_impl &Adapter = ContextImpl.getAdapter();
  Adapter.call<sycl::detail::UrApiKind::urCommandBufferUpdateKernelLaunchExp>(
      CommandBuffer, UpdateDescList.size(), UpdateDescList.data());
}

modifiable_command_graph::modifiable_command_graph(
    const sycl::context &SyclContext, const sycl::device &SyclDevice,
    const sycl::property_list &PropList)
    : impl(std::make_shared<detail::graph_impl>(SyclContext, SyclDevice,
                                                PropList)) {}

modifiable_command_graph::modifiable_command_graph(
    const sycl::queue &SyclQueue, const sycl::property_list &PropList)
    : impl(std::make_shared<detail::graph_impl>(
          SyclQueue.get_context(), SyclQueue.get_device(), PropList)) {}

modifiable_command_graph::modifiable_command_graph(
    const sycl::device &SyclDevice, const sycl::property_list &PropList)
    : impl(std::make_shared<detail::graph_impl>(
          SyclDevice.get_platform().khr_get_default_context(), SyclDevice,
          PropList)) {}

node modifiable_command_graph::addImpl(dynamic_command_group &DynCGF,
                                       const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");
  auto DynCGFImpl = sycl::detail::getSyclObjImpl(DynCGF);

  if (DynCGFImpl->MGraph != impl) {
    throw sycl::exception(make_error_code(sycl::errc::invalid),
                          "Graph does not match the graph associated with "
                          "dynamic command-group.");
  }

  graph_impl::WriteLock Lock(impl->MMutex);
  detail::node_impl &NodeImpl = impl->add(DynCGFImpl, Deps);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

node modifiable_command_graph::addImpl(const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");

  graph_impl::WriteLock Lock(impl->MMutex);
  detail::node_impl &NodeImpl = impl->add(Deps);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

node modifiable_command_graph::addImpl(std::function<void(handler &)> CGF,
                                       const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");

  detail::node_impl &NodeImpl = impl->add(CGF, {}, Deps);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

void modifiable_command_graph::addGraphLeafDependencies(node Node) {
  // Find all exit nodes in the current graph and add them to the dependency
  // vector
  detail::node_impl &DstImpl = *sycl::detail::getSyclObjImpl(Node);
  graph_impl::WriteLock Lock(impl->MMutex);
  for (auto &NodeImpl : impl->MNodeStorage) {
    if ((NodeImpl->MSuccessors.size() == 0) && (NodeImpl.get() != &DstImpl)) {
      impl->makeEdge(*NodeImpl, DstImpl);
    }
  }
}

void modifiable_command_graph::make_edge(node &Src, node &Dest) {
  detail::node_impl &SenderImpl = *sycl::detail::getSyclObjImpl(Src);
  detail::node_impl &ReceiverImpl = *sycl::detail::getSyclObjImpl(Dest);

  graph_impl::WriteLock Lock(impl->MMutex);
  impl->makeEdge(SenderImpl, ReceiverImpl);
}

command_graph<graph_state::executable>
modifiable_command_graph::finalize(const sycl::property_list &PropList) const {
  // Graph is read and written in this scope so we lock
  // this graph with full priviledges.
  graph_impl::WriteLock Lock(impl->MMutex);
  // If the graph uses graph-owned allocations and an executable graph already
  // exists we must throw an error.
  if (impl->getMemPool().hasAllocations() && impl->getExecGraphCount() > 0) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Graphs containing allocations can only have a "
                          "single executable graph alive at any one time.");
  }

  return command_graph<graph_state::executable>{
      this->impl, this->impl->getContext(), PropList};
}

void modifiable_command_graph::begin_recording(
    queue &RecordingQueue, const sycl::property_list &PropList) {
  // No properties is handled here originally, just check that properties are
  // related to graph at all.
  checkGraphPropertiesAndThrow(PropList);

  queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(RecordingQueue);

  if (QueueImpl.hasCommandGraph()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording cannot be called for a queue which "
                          "is already in the recording state.");
  }

  if (QueueImpl.get_context() != impl->getContext()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose context "
                          "differs from the graph context.");
  }
  if (QueueImpl.get_device() != impl->getDevice()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose device "
                          "differs from the graph device.");
  }

  impl->beginRecording(QueueImpl);
}

void modifiable_command_graph::begin_recording(
    const std::vector<queue> &RecordingQueues,
    const sycl::property_list &PropList) {
  for (queue Queue : RecordingQueues) {
    this->begin_recording(Queue, PropList);
  }
}

void modifiable_command_graph::end_recording() {
  impl->clearQueues(true /*Needs lock*/);
}

void modifiable_command_graph::end_recording(queue &RecordingQueue) {
  queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl.getCommandGraph() == impl) {
    QueueImpl.setCommandGraph(nullptr);
    graph_impl::WriteLock Lock(impl->MMutex);
    impl->removeQueue(QueueImpl);
  }
  if (QueueImpl.hasCommandGraph())
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "end_recording called for a queue which is recording "
                          "to a different graph.");
}

void modifiable_command_graph::end_recording(
    const std::vector<queue> &RecordingQueues) {
  for (queue Queue : RecordingQueues) {
    this->end_recording(Queue);
  }
}

void modifiable_command_graph::print_graph(sycl::detail::string_view pathstr,
                                           bool verbose) const {
  std::string path{std::string_view(pathstr)};
  graph_impl::ReadLock Lock(impl->MMutex);
  if (path.substr(path.find_last_of(".") + 1) == "dot") {
    impl->printGraphAsDot(std::move(path), verbose);
  } else {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "DOT graph is the only format supported at the moment.");
  }
}

std::vector<node> modifiable_command_graph::get_nodes() const {
  graph_impl::ReadLock Lock(impl->MMutex);
  return impl->nodes().to<std::vector<node>>();
}
std::vector<node> modifiable_command_graph::get_root_nodes() const {
  graph_impl::ReadLock Lock(impl->MMutex);
  return impl->roots().to<std::vector<node>>();
}

void modifiable_command_graph::checkNodePropertiesAndThrow(
    const property_list &Properties) {
  auto CheckDataLessProperties = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
    switch (PropertyKind) {
#include <sycl/ext/oneapi/experimental/detail/properties/node_properties.def>
    default:
      return false;
    }
  };
  auto CheckPropertiesWithData = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
    switch (PropertyKind) {
#include <sycl/ext/oneapi/experimental/detail/properties/node_properties.def>
    default:
      return false;
    }
  };
  sycl::detail::PropertyValidator::checkPropsAndThrow(
      Properties, CheckDataLessProperties, CheckPropertiesWithData);
}

executable_command_graph::executable_command_graph(
    const std::shared_ptr<detail::graph_impl> &Graph, const sycl::context &Ctx,
    const property_list &PropList)
    : impl(std::make_shared<detail::exec_graph_impl>(Ctx, Graph, PropList)) {
  finalizeImpl(); // Create backend representation for executable graph
  // Mark that we have created an executable graph from the modifiable graph.
  Graph->markExecGraphCreated();
}

void executable_command_graph::finalizeImpl() {
  impl->makePartitions();

  // Handle any work required for graph-owned memory allocations
  impl->finalizeMemoryAllocations();

  auto Device = impl->getGraphImpl()->getDevice();
  for (auto Partition : impl->getPartitions()) {
    if (!Partition->MIsHostTask) {
      impl->createCommandBuffers(Device, Partition);
    }
  }
  impl->buildRequirements();
}

void executable_command_graph::update(
    const command_graph<graph_state::modifiable> &Graph) {
  impl->update(sycl::detail::getSyclObjImpl(Graph));
}

void executable_command_graph::update(const node &Node) {
  impl->update(*sycl::detail::getSyclObjImpl(Node));
}

void executable_command_graph::update(const std::vector<node> &Nodes) {
  impl->update(Nodes);
}

size_t executable_command_graph::get_required_mem_size() const {
  // Since each graph has a unique mem pool, return the current memory usage for
  // now. This call my change if we move to being able to share memory between
  // unique graphs.
  return impl->getGraphImpl()->getMemPool().getMemUseCurrent();
}
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
