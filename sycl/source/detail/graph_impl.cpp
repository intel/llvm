//==--------- graph_impl.cpp - SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/graph_impl.hpp>
#include <detail/handler_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

namespace {
/// Visits a node on the graph and it's successors recursively in a depth-first
/// approach.
/// @param[in] Node The current node being visited.
/// @param[in,out] VisitedNodes A set of unique nodes which have already been
/// visited.
/// @param[in] NodeStack Stack of nodes which are currently being visited on the
/// current path through the graph.
/// @param[in] NodeFunc The function object to be run on each node. A return
/// value of true indicates the search should be ended immediately and the
/// function will return.
/// @return True if the search should end immediately, false if not.
bool visitNodeDepthFirst(
    std::shared_ptr<node_impl> Node,
    std::set<std::shared_ptr<node_impl>> &VisitedNodes,
    std::deque<std::shared_ptr<node_impl>> &NodeStack,
    std::function<bool(std::shared_ptr<node_impl> &,
                       std::deque<std::shared_ptr<node_impl>> &)>
        NodeFunc) {
  auto EarlyReturn = NodeFunc(Node, NodeStack);
  if (EarlyReturn) {
    return true;
  }
  NodeStack.push_back(Node);
  Node->MVisited = true;
  VisitedNodes.emplace(Node);
  for (auto &Successor : Node->MSuccessors) {
    if (visitNodeDepthFirst(Successor.lock(), VisitedNodes, NodeStack,
                            NodeFunc)) {
      return true;
    }
  }
  NodeStack.pop_back();
  return false;
}

/// Recursively add nodes to execution stack.
/// @param NodeImpl Node to schedule.
/// @param Schedule Execution ordering to add node to.
/// @param PartitionBounded If set to true, the topological sort is stopped at
/// partition borders. Hence, nodes belonging to a partition different from the
/// NodeImpl partition are not processed.
void sortTopological(std::shared_ptr<node_impl> NodeImpl,
                     std::list<std::shared_ptr<node_impl>> &Schedule,
                     bool PartitionBounded = false) {
  for (auto &Succ : NodeImpl->MSuccessors) {
    auto NextNode = Succ.lock();
    if (PartitionBounded &&
        (NextNode->MPartitionNum != NodeImpl->MPartitionNum)) {
      continue;
    }
    // Check if we've already scheduled this node
    if (std::find(Schedule.begin(), Schedule.end(), NextNode) ==
        Schedule.end()) {
      sortTopological(NextNode, Schedule, PartitionBounded);
    }
  }

  Schedule.push_front(NodeImpl);
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
void propagatePartitionUp(std::shared_ptr<node_impl> Node, int PartitionNum) {
  if (((Node->MPartitionNum != -1) && (Node->MPartitionNum <= PartitionNum)) ||
      (Node->MCGType == sycl::detail::CG::CGTYPE::CodeplayHostTask)) {
    return;
  }
  Node->MPartitionNum = PartitionNum;
  for (auto &Predecessor : Node->MPredecessors) {
    propagatePartitionUp(Predecessor.lock(), PartitionNum);
  }
}

/// Propagates the partition number `PartitionNum` to successors.
/// Propagation stops when an host task is encountered or when no successors
/// remain.
/// @param Node Node to assign to the partition.
/// @param PartitionNum Number to propagate.
/// @param HostTaskList List of host tasks that have already been processed and
/// are encountered as successors to the node Node.
void propagatePartitionDown(
    std::shared_ptr<node_impl> Node, int PartitionNum,
    std::list<std::shared_ptr<node_impl>> &HostTaskList) {
  if (Node->MCGType == sycl::detail::CG::CGTYPE::CodeplayHostTask) {
    if (Node->MPartitionNum != -1) {
      HostTaskList.push_front(Node);
    }
    return;
  }
  Node->MPartitionNum = PartitionNum;
  for (auto &Successor : Node->MSuccessors) {
    propagatePartitionDown(Successor.lock(), PartitionNum, HostTaskList);
  }
}

/// Tests if the node is a root of its partition (i.e. no predecessors that
/// belong to the same partition)
/// @param Node node to test
/// @return True is `Node` is a root of its partition
bool isPartitionRoot(std::shared_ptr<node_impl> Node) {
  for (auto &Predecessor : Node->MPredecessors) {
    if (Predecessor.lock()->MPartitionNum == Node->MPartitionNum) {
      return false;
    }
  }
  return true;
}

/// Takes a vector of weak_ptrs to node_impls and returns a vector of node
/// objects created from those impls, in the same order.
std::vector<node> createNodesFromImpls(
    const std::vector<std::weak_ptr<detail::node_impl>> &Impls) {
  std::vector<node> Nodes{};
  Nodes.reserve(Impls.size());

  for (std::weak_ptr<detail::node_impl> Impl : Impls) {
    Nodes.push_back(sycl::detail::createSyclObjFromImpl<node>(Impl.lock()));
  }

  return Nodes;
}

/// Takes a vector of shared_ptrs to node_impls and returns a vector of node
/// objects created from those impls, in the same order.
std::vector<node> createNodesFromImpls(
    const std::vector<std::shared_ptr<detail::node_impl>> &Impls) {
  std::vector<node> Nodes{};
  Nodes.reserve(Impls.size());

  for (std::shared_ptr<detail::node_impl> Impl : Impls) {
    Nodes.push_back(sycl::detail::createSyclObjFromImpl<node>(Impl));
  }

  return Nodes;
}

} // anonymous namespace

void partition::schedule() {
  if (MSchedule.empty()) {
    for (auto &Node : MRoots) {
      sortTopological(Node.lock(), MSchedule, true);
    }
  }
}

void exec_graph_impl::makePartitions() {
  int CurrentPartition = -1;
  std::list<std::shared_ptr<node_impl>> HostTaskList;
  // find all the host-tasks in the graph
  for (auto &Node : MNodeStorage) {
    if (Node->MCGType == sycl::detail::CG::CodeplayHostTask) {
      HostTaskList.push_back(Node);
    }
  }

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
    auto Node = HostTaskList.front();
    HostTaskList.pop_front();
    CurrentPartition++;
    for (auto &Predecessor : Node->MPredecessors) {
      propagatePartitionUp(Predecessor.lock(), CurrentPartition);
    }
    CurrentPartition++;
    Node->MPartitionNum = CurrentPartition;
    CurrentPartition++;
    auto TmpSize = HostTaskList.size();
    for (auto &Successor : Node->MSuccessors) {
      propagatePartitionDown(Successor.lock(), CurrentPartition, HostTaskList);
    }
    if (HostTaskList.size() > TmpSize) {
      // At least one HostTask has been re-numbered so group merge opportunities
      for (const auto &HT : HostTaskList) {
        auto HTPartitionNum = HT->MPartitionNum;
        if (HTPartitionNum != -1) {
          // can merge predecessors of node `Node` with predecessors of node
          // `HT` (HTPartitionNum-1) since HT must be reprocessed
          for (const auto &NodeImpl : MNodeStorage) {
            if (NodeImpl->MPartitionNum == Node->MPartitionNum - 1) {
              NodeImpl->MPartitionNum = HTPartitionNum - 1;
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
    for (auto &Node : MNodeStorage) {
      if (Node->MPartitionNum == i) {
        MPartitionNodes[Node] = PartitionFinalNum;
        if (isPartitionRoot(Node)) {
          Partition->MRoots.insert(Node);
        }
      }
    }
    if (Partition->MRoots.size() > 0) {
      Partition->schedule();
      Partition->MIsInOrderGraph = Partition->checkIfGraphIsSinglePath();
      MPartitions.push_back(Partition);
      PartitionFinalNum++;
    }
  }

  // Add an empty partition if there is no partition, i.e. empty graph
  if (MPartitions.size() == 0) {
    MPartitions.push_back(std::make_shared<partition>());
  }

  // Make global schedule list
  for (const auto &Partition : MPartitions) {
    MSchedule.insert(MSchedule.end(), Partition->MSchedule.begin(),
                     Partition->MSchedule.end());
  }

  // Compute partition dependencies
  for (const auto &Partition : MPartitions) {
    for (auto const &Root : Partition->MRoots) {
      auto RootNode = Root.lock();
      for (const auto &Dep : RootNode->MPredecessors) {
        auto NodeDep = Dep.lock();
        Partition->MPredecessors.push_back(
            MPartitions[MPartitionNodes[NodeDep]]);
      }
    }
  }

  // Reset node groups (if node have to be re-processed - e.g. subgraph)
  for (auto &Node : MNodeStorage) {
    Node->MPartitionNum = -1;
  }
}

graph_impl::~graph_impl() {
  try {
    clearQueues();
    for (auto &MemObj : MMemObjs) {
      MemObj->markNoLongerBeingUsedInGraph();
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~graph_impl", e);
  }
}

std::shared_ptr<node_impl> graph_impl::addNodesToExits(
    const std::shared_ptr<graph_impl> &Impl,
    const std::list<std::shared_ptr<node_impl>> &NodeList) {
  // Find all input and output nodes from the node list
  std::vector<std::shared_ptr<node_impl>> Inputs;
  std::vector<std::shared_ptr<node_impl>> Outputs;
  for (auto &NodeImpl : NodeList) {
    if (NodeImpl->MPredecessors.size() == 0) {
      Inputs.push_back(NodeImpl);
    }
    if (NodeImpl->MSuccessors.size() == 0) {
      Outputs.push_back(NodeImpl);
    }
  }

  // Find all exit nodes in the current graph and register the Inputs as
  // successors
  for (auto &NodeImpl : MNodeStorage) {
    if (NodeImpl->MSuccessors.size() == 0) {
      for (auto &Input : Inputs) {
        NodeImpl->registerSuccessor(Input, NodeImpl);
      }
    }
  }

  // Add all the new nodes to the node storage
  for (auto &Node : NodeList) {
    MNodeStorage.push_back(Node);
    addEventForNode(Impl, std::make_shared<sycl::detail::event_impl>(), Node);
  }

  return this->add(Impl, Outputs);
}

void graph_impl::addRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.insert(Root);
}

void graph_impl::removeRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.erase(Root);
}

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &Impl,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
  // Copy deps so we can modify them
  auto Deps = Dep;

  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>();

  MNodeStorage.push_back(NodeImpl);

  addDepsToNode(NodeImpl, Deps);
  // Add an event associated with this explicit node for mixed usage
  addEventForNode(Impl, std::make_shared<sycl::detail::event_impl>(), NodeImpl);
  return NodeImpl;
}

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &Impl,
                std::function<void(handler &)> CGF,
                const std::vector<sycl::detail::ArgDesc> &Args,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
  (void)Args;
  sycl::handler Handler{Impl};
  CGF(Handler);

  if (Handler.MCGType == sycl::detail::CG::Barrier) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "The sycl_ext_oneapi_enqueue_barrier feature is not available with "
        "SYCL Graph Explicit API. Please use empty nodes instead.");
  }

  Handler.finalize();

  node_type NodeType =
      Handler.MImpl->MUserFacingNodeType !=
              ext::oneapi::experimental::node_type::empty
          ? Handler.MImpl->MUserFacingNodeType
          : ext::oneapi::experimental::detail::getNodeTypeFromCG(
                Handler.MCGType);

  auto NodeImpl = this->add(NodeType, std::move(Handler.MGraphNodeCG), Dep);
  NodeImpl->MNDRangeUsed = Handler.MImpl->MNDRangeUsed;
  // Add an event associated with this explicit node for mixed usage
  addEventForNode(Impl, std::make_shared<sycl::detail::event_impl>(), NodeImpl);

  // Retrieve any dynamic parameters which have been registered in the CGF and
  // register the actual nodes with them.
  auto &DynamicParams = Handler.MImpl->MDynamicParameters;

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

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &Impl,
                const std::vector<sycl::detail::EventImplPtr> Events) {

  std::vector<std::shared_ptr<node_impl>> Deps;

  // Add any nodes specified by event dependencies into the dependency list
  for (const auto &Dep : Events) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      Deps.push_back(NodeImpl->second);
    } else {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    }
  }

  return this->add(Impl, Deps);
}

std::shared_ptr<node_impl>
graph_impl::add(node_type NodeType,
                std::unique_ptr<sycl::detail::CG> CommandGroup,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
  // Copy deps so we can modify them
  auto Deps = Dep;

  // A unique set of dependencies obtained by checking requirements and events
  std::set<std::shared_ptr<node_impl>> UniqueDeps;
  const auto &Requirements = CommandGroup->getRequirements();
  if (!MAllowBuffers && Requirements.size()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Cannot use buffers in a graph without passing the "
                          "assume_buffer_outlives_graph property on "
                          "Graph construction.");
  }

  for (auto &Req : Requirements) {
    // Track and mark the memory objects being used by the graph.
    auto MemObj = static_cast<sycl::detail::SYCLMemObjT *>(Req->MSYCLMemObj);
    bool WasInserted = MMemObjs.insert(MemObj).second;
    if (WasInserted) {
      MemObj->markBeingUsedInGraph();
    }
    // Look through the graph for nodes which share this requirement
    for (auto &Node : MNodeStorage) {
      if (Node->hasRequirementDependency(Req)) {
        bool ShouldAddDep = true;
        // If any of this node's successors have this requirement then we skip
        // adding the current node as a dependency.
        for (auto &Succ : Node->MSuccessors) {
          if (Succ.lock()->hasRequirementDependency(Req)) {
            ShouldAddDep = false;
            break;
          }
        }
        if (ShouldAddDep) {
          UniqueDeps.insert(Node);
        }
      }
    }
  }

  // Add any nodes specified by event dependencies into the dependency list
  for (auto &Dep : CommandGroup->getEvents()) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      UniqueDeps.insert(NodeImpl->second);
    } else {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    }
  }
  // Add any deps determined from requirements and events into the dependency
  // list
  Deps.insert(Deps.end(), UniqueDeps.begin(), UniqueDeps.end());

  const std::shared_ptr<node_impl> &NodeImpl =
      std::make_shared<node_impl>(NodeType, std::move(CommandGroup));
  MNodeStorage.push_back(NodeImpl);

  addDepsToNode(NodeImpl, Deps);

  return NodeImpl;
}

bool graph_impl::clearQueues() {
  bool AnyQueuesCleared = false;
  for (auto &Queue : MRecordingQueues) {
    if (auto ValidQueue = Queue.lock(); ValidQueue) {
      ValidQueue->setCommandGraph(nullptr);
      AnyQueuesCleared = true;
    }
  }
  MRecordingQueues.clear();

  return AnyQueuesCleared;
}

void graph_impl::searchDepthFirst(
    std::function<bool(std::shared_ptr<node_impl> &,
                       std::deque<std::shared_ptr<node_impl>> &)>
        NodeFunc) {
  // Track nodes visited during the search which can be used by NodeFunc in
  // depth first search queries. Currently unusued but is an
  // integral part of depth first searches.
  std::set<std::shared_ptr<node_impl>> VisitedNodes;

  for (auto &Root : MRoots) {
    std::deque<std::shared_ptr<node_impl>> NodeStack;
    if (visitNodeDepthFirst(Root.lock(), VisitedNodes, NodeStack, NodeFunc)) {
      break;
    }
  }

  // Reset the visited status of all nodes encountered in the search.
  for (auto &Node : VisitedNodes) {
    Node->MVisited = false;
  }
}

bool graph_impl::checkForCycles() {
  // Using a depth-first search and checking if we vist a node more than once in
  // the current path to identify if there are cycles.
  bool CycleFound = false;
  auto CheckFunc = [&](std::shared_ptr<node_impl> &Node,
                       std::deque<std::shared_ptr<node_impl>> &NodeStack) {
    // If the current node has previously been found in the current path through
    // the graph then we have a cycle and we end the search early.
    if (std::find(NodeStack.begin(), NodeStack.end(), Node) !=
        NodeStack.end()) {
      CycleFound = true;
      return true;
    }
    return false;
  };
  searchDepthFirst(CheckFunc);
  return CycleFound;
}

void graph_impl::makeEdge(std::shared_ptr<node_impl> Src,
                          std::shared_ptr<node_impl> Dest) {
  throwIfGraphRecordingQueue("make_edge()");
  if (Src == Dest) {
    throw sycl::exception(
        make_error_code(sycl::errc::invalid),
        "make_edge() cannot be called when Src and Dest are the same.");
  }

  bool SrcFound = false;
  bool DestFound = false;
  for (const auto &Node : MNodeStorage) {

    SrcFound |= Node == Src;
    DestFound |= Node == Dest;

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

  // We need to add the edges first before checking for cycles
  Src->registerSuccessor(Dest, Src);

  // We can skip cycle checks if either Dest has no successors (cycle not
  // possible) or cycle checks have been disabled with the no_cycle_check
  // property;
  if (Dest->MSuccessors.empty() || !MSkipCycleChecks) {
    bool CycleFound = checkForCycles();

    if (CycleFound) {
      // Remove the added successor and predecessor
      Src->MSuccessors.pop_back();
      Dest->MPredecessors.pop_back();

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
  for (auto &Node : MNodeStorage) {
    if (Node->MSuccessors.empty()) {
      auto EventForNode = getEventForNode(Node);
      if (EventForNode->getSubmittedQueue() == RecordedQueueSP) {
        Events.push_back(getEventForNode(Node));
      }
    }
  }

  return Events;
}

// Check if nodes are empty and if so loop back through predecessors until we
// find the real dependency.
void exec_graph_impl::findRealDeps(
    std::vector<sycl::detail::pi::PiExtSyncPoint> &Deps,
    std::shared_ptr<node_impl> CurrentNode, int ReferencePartitionNum) {
  if (CurrentNode->isEmpty()) {
    for (auto &N : CurrentNode->MPredecessors) {
      auto NodeImpl = N.lock();
      findRealDeps(Deps, NodeImpl, ReferencePartitionNum);
    }
  } else {
    // Verify if CurrentNode belong the the same partition
    if (MPartitionNodes[CurrentNode] == ReferencePartitionNum) {
      // Verify that the sync point has actually been set for this node.
      auto SyncPoint = MPiSyncPoints.find(CurrentNode);
      assert(SyncPoint != MPiSyncPoints.end() &&
             "No sync point has been set for node dependency.");
      // Check if the dependency has already been added.
      if (std::find(Deps.begin(), Deps.end(), SyncPoint->second) ==
          Deps.end()) {
        Deps.push_back(SyncPoint->second);
      }
    }
  }
}

sycl::detail::pi::PiExtSyncPoint exec_graph_impl::enqueueNodeDirect(
    sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    std::shared_ptr<node_impl> Node) {
  std::vector<sycl::detail::pi::PiExtSyncPoint> Deps;
  for (auto &N : Node->MPredecessors) {
    findRealDeps(Deps, N.lock(), MPartitionNodes[Node]);
  }
  sycl::detail::pi::PiExtSyncPoint NewSyncPoint;
  sycl::detail::pi::PiExtCommandBufferCommand NewCommand = 0;
  pi_int32 Res = sycl::detail::enqueueImpCommandBufferKernel(
      Ctx, DeviceImpl, CommandBuffer,
      *static_cast<sycl::detail::CGExecKernel *>((Node->MCommandGroup.get())),
      Deps, &NewSyncPoint, &NewCommand, nullptr);

  MCommandMap[Node] = NewCommand;

  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::invalid,
                          "Failed to add kernel to PI command-buffer");
  }

  return NewSyncPoint;
}

sycl::detail::pi::PiExtSyncPoint exec_graph_impl::enqueueNode(
    sycl::context Ctx, std::shared_ptr<sycl::detail::device_impl> DeviceImpl,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    std::shared_ptr<node_impl> Node) {

  // Queue which will be used for allocation operations for accessors.
  auto AllocaQueue = std::make_shared<sycl::detail::queue_impl>(
      DeviceImpl, sycl::detail::getSyclObjImpl(Ctx), sycl::async_handler{},
      sycl::property_list{});

  std::vector<sycl::detail::pi::PiExtSyncPoint> Deps;
  for (auto &N : Node->MPredecessors) {
    findRealDeps(Deps, N.lock(), MPartitionNodes[Node]);
  }

  sycl::detail::EventImplPtr Event =
      sycl::detail::Scheduler::getInstance().addCG(
          Node->getCGCopy(), AllocaQueue, /*EventNeeded=*/true, CommandBuffer,
          Deps);

  MCommandMap[Node] = Event->getCommandBufferCommand();
  return Event->getSyncPoint();
}
void exec_graph_impl::createCommandBuffers(
    sycl::device Device, std::shared_ptr<partition> &Partition) {
  sycl::detail::pi::PiExtCommandBuffer OutCommandBuffer;
  sycl::detail::pi::PiExtCommandBufferDesc Desc{
      pi_ext_structure_type::PI_EXT_STRUCTURE_TYPE_COMMAND_BUFFER_DESC, nullptr,
      pi_bool(Partition->MIsInOrderGraph && !MEnableProfiling),
      pi_bool(MEnableProfiling), pi_bool(MIsUpdatable)};

  auto ContextImpl = sycl::detail::getSyclObjImpl(MContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  auto DeviceImpl = sycl::detail::getSyclObjImpl(Device);
  pi_result Res =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextCommandBufferCreate>(
          ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(), &Desc,
          &OutCommandBuffer);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::invalid, "Failed to create PI command-buffer");
  }

  Partition->MPiCommandBuffers[Device] = OutCommandBuffer;

  for (const auto &Node : Partition->MSchedule) {
    // Empty nodes are not processed as other nodes, but only their
    // dependencies are propagated in findRealDeps
    if (Node->isEmpty())
      continue;

    sycl::detail::CG::CGTYPE type = Node->MCGType;
    // If the node is a kernel with no special requirements we can enqueue it
    // directly.
    if (type == sycl::detail::CG::Kernel &&
        Node->MCommandGroup->getRequirements().size() +
                static_cast<sycl::detail::CGExecKernel *>(
                    Node->MCommandGroup.get())
                    ->MStreams.size() ==
            0) {
      MPiSyncPoints[Node] =
          enqueueNodeDirect(MContext, DeviceImpl, OutCommandBuffer, Node);
    } else {
      MPiSyncPoints[Node] =
          enqueueNode(MContext, DeviceImpl, OutCommandBuffer, Node);
    }

    // Append Node requirements to overall graph requirements
    MRequirements.insert(MRequirements.end(),
                         Node->MCommandGroup->getRequirements().begin(),
                         Node->MCommandGroup->getRequirements().end());
    // Also store the actual accessor to make sure they are kept alive when
    // commands are submitted
    MAccessors.insert(MAccessors.end(),
                      Node->MCommandGroup->getAccStorage().begin(),
                      Node->MCommandGroup->getAccStorage().end());
  }

  Res =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextCommandBufferFinalize>(
          OutCommandBuffer);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::invalid,
                          "Failed to finalize PI command-buffer");
  }
}

exec_graph_impl::exec_graph_impl(sycl::context Context,
                                 const std::shared_ptr<graph_impl> &GraphImpl,
                                 const property_list &PropList)
    : MSchedule(), MGraphImpl(GraphImpl), MPiSyncPoints(),
      MDevice(GraphImpl->getDevice()), MContext(Context), MRequirements(),
      MExecutionEvents(),
      MIsUpdatable(PropList.has_property<property::graph::updatable>()),
      MEnableProfiling(
          PropList.has_property<property::graph::enable_profiling>()) {

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
}

exec_graph_impl::~exec_graph_impl() {
  try {
    const sycl::detail::PluginPtr &Plugin =
        sycl::detail::getSyclObjImpl(MContext)->getPlugin();
    MSchedule.clear();
    // We need to wait on all command buffer executions before we can release
    // them.
    for (auto &Event : MExecutionEvents) {
      Event->wait(Event);
    }

    for (const auto &Partition : MPartitions) {
      Partition->MSchedule.clear();
      for (const auto &Iter : Partition->MPiCommandBuffers) {
        if (auto CmdBuf = Iter.second; CmdBuf) {
          pi_result Res = Plugin->call_nocheck<
              sycl::detail::PiApiKind::piextCommandBufferRelease>(CmdBuf);
          (void)Res;
          assert(Res == pi_result::PI_SUCCESS);
        }
      }
    }

    for (auto &Iter : MCommandMap) {
      if (auto Command = Iter.second; Command) {
        pi_result Res = Plugin->call_nocheck<
            sycl::detail::PiApiKind::piextCommandBufferReleaseCommand>(Command);
        (void)Res;
        assert(Res == pi_result::PI_SUCCESS);
      }
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~exec_graph_impl", e);
  }
}

sycl::event
exec_graph_impl::enqueue(const std::shared_ptr<sycl::detail::queue_impl> &Queue,
                         sycl::detail::CG::StorageInitHelper CGData) {
  WriteLock Lock(MMutex);

  // Map of the partitions to their execution events
  std::unordered_map<std::shared_ptr<partition>, sycl::detail::EventImplPtr>
      PartitionsExecutionEvents;

  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });

  sycl::detail::EventImplPtr NewEvent;
  std::vector<sycl::detail::EventImplPtr> BackupCGDataMEvents;
  if (MPartitions.size() > 1) {
    BackupCGDataMEvents = CGData.MEvents;
  }
  for (uint32_t currentPartitionsNum = 0;
       currentPartitionsNum < MPartitions.size(); currentPartitionsNum++) {
    auto CurrentPartition = MPartitions[currentPartitionsNum];
    // restore initial MEvents to add only needed additional depenencies
    if (currentPartitionsNum > 0) {
      CGData.MEvents = BackupCGDataMEvents;
    }

    for (auto const &DepPartition : CurrentPartition->MPredecessors) {
      CGData.MEvents.push_back(PartitionsExecutionEvents[DepPartition]);
    }

    auto CommandBuffer =
        CurrentPartition->MPiCommandBuffers[Queue->get_device()];

    if (CommandBuffer) {
      // if previous submissions are incompleted, we automatically
      // add completion events of previous submissions as dependencies.
      // With Level-Zero backend we cannot resubmit a command-buffer until the
      // previous one has already completed.
      // Indeed, since a command-list does not accept a list a dependencies at
      // submission, we circumvent this lack by adding a barrier that waits on a
      // specific event and then define the conditions to signal this event in
      // another command-list. Consequently, if a second submission is
      // performed, the signal conditions of this single event are redefined by
      // this second submission. Thus, this can lead to an undefined behaviour
      // and potential hangs. We have therefore to expliclty wait in the host
      // for previous submission to complete before resubmitting the
      // command-buffer for level-zero backend.
      // TODO : add a check to release this constraint and allow multiple
      // concurrent submissions if the exec_graph has been updated since the
      // last submission.
      for (std::vector<sycl::detail::EventImplPtr>::iterator It =
               MExecutionEvents.begin();
           It != MExecutionEvents.end();) {
        auto Event = *It;
        if (!Event->isCompleted()) {
          if (Queue->get_device().get_backend() ==
              sycl::backend::ext_oneapi_level_zero) {
            Event->wait(Event);
          } else {
            auto &AttachedEventsList = Event->getPostCompleteEvents();
            CGData.MEvents.reserve(AttachedEventsList.size() + 1);
            CGData.MEvents.push_back(Event);
            // Add events of the previous execution of all graph partitions.
            for (auto &AttachedEvent : AttachedEventsList) {
              CGData.MEvents.push_back(AttachedEvent);
            }
          }
          ++It;
        } else {
          // Remove completed events
          It = MExecutionEvents.erase(It);
        }
      }

      NewEvent = CreateNewEvent();
      sycl::detail::pi::PiEvent *OutEvent = &NewEvent->getHandleRef();
      // Merge requirements from the nodes into requirements (if any) from the
      // handler.
      CGData.MRequirements.insert(CGData.MRequirements.end(),
                                  MRequirements.begin(), MRequirements.end());
      CGData.MAccStorage.insert(CGData.MAccStorage.end(), MAccessors.begin(),
                                MAccessors.end());

      // If we have no requirements or dependent events for the command buffer,
      // enqueue it directly
      if (CGData.MRequirements.empty() && CGData.MEvents.empty()) {
        if (NewEvent != nullptr)
          NewEvent->setHostEnqueueTime();
        pi_result Res =
            Queue->getPlugin()
                ->call_nocheck<
                    sycl::detail::PiApiKind::piextEnqueueCommandBuffer>(
                    CommandBuffer, Queue->getHandleRef(), 0, nullptr, OutEvent);
        if (Res == pi_result::PI_ERROR_INVALID_QUEUE_PROPERTIES) {
          throw sycl::exception(
              make_error_code(errc::invalid),
              "Graphs cannot be submitted to a queue which uses "
              "immediate command lists. Use "
              "sycl::ext::intel::property::queue::no_immediate_"
              "command_list to disable them.");
        } else if (Res != pi_result::PI_SUCCESS) {
          throw sycl::exception(
              errc::event,
              "Failed to enqueue event for command buffer submission");
        }
      } else {
        std::unique_ptr<sycl::detail::CG> CommandGroup =
            std::make_unique<sycl::detail::CGExecCommandBuffer>(
                CommandBuffer, nullptr, std::move(CGData));

        NewEvent = sycl::detail::Scheduler::getInstance().addCG(
            std::move(CommandGroup), Queue, /*EventNeeded=*/true);
      }
      NewEvent->setEventFromSubmittedExecCommandBuffer(true);
    } else if ((CurrentPartition->MSchedule.size() > 0) &&
               (CurrentPartition->MSchedule.front()->MCGType ==
                sycl::detail::CG::CGTYPE::CodeplayHostTask)) {
      auto NodeImpl = CurrentPartition->MSchedule.front();
      // Schedule host task
      NodeImpl->MCommandGroup->getEvents().insert(
          NodeImpl->MCommandGroup->getEvents().end(), CGData.MEvents.begin(),
          CGData.MEvents.end());
      // HostTask CG stores the Queue on which the task was submitted.
      // In case of graph, this queue may differ from the actual execution
      // queue. We therefore overload this Queue before submitting the task.
      static_cast<sycl::detail::CGHostTask &>(*NodeImpl->MCommandGroup.get())
          .MQueue = Queue;

      NewEvent = sycl::detail::Scheduler::getInstance().addCG(
          NodeImpl->getCGCopy(), Queue, /*EventNeeded=*/true);
    } else {
      std::vector<std::shared_ptr<sycl::detail::event_impl>> ScheduledEvents;
      for (auto &NodeImpl : CurrentPartition->MSchedule) {
        std::vector<sycl::detail::pi::PiEvent> RawEvents;

        // If the node has no requirements for accessors etc. then we skip the
        // scheduler and enqueue directly.
        if (NodeImpl->MCGType == sycl::detail::CG::Kernel &&
            NodeImpl->MCommandGroup->getRequirements().size() +
                    static_cast<sycl::detail::CGExecKernel *>(
                        NodeImpl->MCommandGroup.get())
                        ->MStreams.size() ==
                0) {
          sycl::detail::CGExecKernel *CG =
              static_cast<sycl::detail::CGExecKernel *>(
                  NodeImpl->MCommandGroup.get());
          auto OutEvent = CreateNewEvent();
          pi_int32 Res = sycl::detail::enqueueImpKernel(
              Queue, CG->MNDRDesc, CG->MArgs, CG->MKernelBundle,
              CG->MSyclKernel, CG->MKernelName, RawEvents, OutEvent,
              // TODO: Pass accessor mem allocations
              nullptr,
              // TODO: Extract from handler
              PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT, CG->MKernelIsCooperative);
          if (Res != pi_result::PI_SUCCESS) {
            throw sycl::exception(
                sycl::make_error_code(sycl::errc::kernel),
                "Error during emulated graph command group submission.");
          }
          ScheduledEvents.push_back(NewEvent);
        } else if (!NodeImpl->isEmpty()) {
          // Empty nodes are node processed as other nodes, but only their
          // dependencies are propagated in findRealDeps
          sycl::detail::EventImplPtr EventImpl =
              sycl::detail::Scheduler::getInstance().addCG(
                  NodeImpl->getCGCopy(), Queue, /*EventNeeded=*/true);

          ScheduledEvents.push_back(EventImpl);
        }
      }
      // Create an event which has all kernel events as dependencies
      NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
      NewEvent->setStateIncomplete();
      NewEvent->getPreparedDepsEvents() = ScheduledEvents;
    }
    PartitionsExecutionEvents[CurrentPartition] = NewEvent;
  }

  // Keep track of this execution event so we can make sure it's completed in
  // the destructor.
  MExecutionEvents.push_back(NewEvent);
  // Attach events of previous partitions to ensure that when the returned event
  // is complete all execution associated with the graph have been completed.
  for (auto const &Elem : PartitionsExecutionEvents) {
    if (Elem.second != NewEvent) {
      NewEvent->attachEventToComplete(Elem.second);
    }
  }
  NewEvent->setProfilingEnabled(MEnableProfiling);
  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  return QueueEvent;
}

void exec_graph_impl::duplicateNodes() {
  // Map of original modifiable nodes (keys) to new duplicated nodes (values)
  std::map<std::shared_ptr<node_impl>, std::shared_ptr<node_impl>> NodesMap;

  const std::vector<std::shared_ptr<node_impl>> &ModifiableNodes =
      MGraphImpl->MNodeStorage;
  std::deque<std::shared_ptr<node_impl>> NewNodes;

  for (size_t i = 0; i < ModifiableNodes.size(); i++) {
    auto OriginalNode = ModifiableNodes[i];
    std::shared_ptr<node_impl> NodeCopy =
        std::make_shared<node_impl>(*OriginalNode);

    // Associate the ID of the original node with the node copy for later quick
    // access
    MIDCache.insert(std::make_pair(OriginalNode->MID, NodeCopy));

    // Clear edges between nodes so that we can replace with new ones
    NodeCopy->MSuccessors.clear();
    NodeCopy->MPredecessors.clear();
    // Push the new node to the front of the stack
    NewNodes.push_back(NodeCopy);
    // Associate the new node with the old one for updating edges
    NodesMap.insert({OriginalNode, NodeCopy});
  }

  // Now that all nodes have been copied rebuild edges on new nodes. This must
  // be done as a separate step since successors may be out of order.
  for (size_t i = 0; i < ModifiableNodes.size(); i++) {
    auto OriginalNode = ModifiableNodes[i];
    auto NodeCopy = NewNodes[i];
    // Look through all the original node successors, find their copies and
    // register those as successors with the current copied node
    for (auto &NextNode : OriginalNode->MSuccessors) {
      auto Successor = NodesMap.at(NextNode.lock());
      NodeCopy->registerSuccessor(Successor, NodeCopy);
    }
  }

  // Subgraph nodes need special handling, we extract all subgraph nodes and
  // merge them into the main node list

  for (auto NewNodeIt = NewNodes.rbegin(); NewNodeIt != NewNodes.rend();
       ++NewNodeIt) {
    auto NewNode = *NewNodeIt;
    if (NewNode->MNodeType != node_type::subgraph) {
      continue;
    }
    const std::vector<std::shared_ptr<node_impl>> &SubgraphNodes =
        NewNode->MSubGraphImpl->MNodeStorage;
    std::deque<std::shared_ptr<node_impl>> NewSubgraphNodes{};

    // Map of original subgraph nodes (keys) to new duplicated nodes (values)
    std::map<std::shared_ptr<node_impl>, std::shared_ptr<node_impl>>
        SubgraphNodesMap;

    // Copy subgraph nodes
    for (size_t i = 0; i < SubgraphNodes.size(); i++) {
      auto SubgraphNode = SubgraphNodes[i];
      auto NodeCopy = std::make_shared<node_impl>(*SubgraphNode);
      // Associate the ID of the original subgraph node with all extracted node
      // copies for future quick access.
      MIDCache.insert(std::make_pair(SubgraphNode->MID, NodeCopy));

      NewSubgraphNodes.push_back(NodeCopy);
      SubgraphNodesMap.insert({SubgraphNode, NodeCopy});
      NodeCopy->MSuccessors.clear();
      NodeCopy->MPredecessors.clear();
    }

    // Rebuild edges for new subgraph nodes
    for (size_t i = 0; i < SubgraphNodes.size(); i++) {
      auto SubgraphNode = SubgraphNodes[i];
      auto NodeCopy = NewSubgraphNodes[i];

      for (auto &NextNode : SubgraphNode->MSuccessors) {
        auto Successor = SubgraphNodesMap.at(NextNode.lock());
        NodeCopy->registerSuccessor(Successor, NodeCopy);
      }
    }

    // Collect input and output nodes for the subgraph
    std::vector<std::shared_ptr<node_impl>> Inputs;
    std::vector<std::shared_ptr<node_impl>> Outputs;
    for (auto &NodeImpl : NewSubgraphNodes) {
      if (NodeImpl->MPredecessors.size() == 0) {
        Inputs.push_back(NodeImpl);
      }
      if (NodeImpl->MSuccessors.size() == 0) {
        Outputs.push_back(NodeImpl);
      }
    }

    // Update the predecessors and successors of the nodes which reference the
    // original subgraph node

    // Predecessors
    for (auto &PredNodeWeak : NewNode->MPredecessors) {
      auto PredNode = PredNodeWeak.lock();
      auto &Successors = PredNode->MSuccessors;

      // Remove the subgraph node from this nodes successors
      Successors.erase(std::remove_if(Successors.begin(), Successors.end(),
                                      [NewNode](auto WeakNode) {
                                        return WeakNode.lock() == NewNode;
                                      }),
                       Successors.end());

      // Add all input nodes from the subgraph as successors for this node
      // instead
      for (auto &Input : Inputs) {
        PredNode->registerSuccessor(Input, PredNode);
      }
    }

    // Successors
    for (auto &SuccNodeWeak : NewNode->MSuccessors) {
      auto SuccNode = SuccNodeWeak.lock();
      auto &Predecessors = SuccNode->MPredecessors;

      // Remove the subgraph node from this nodes successors
      Predecessors.erase(std::remove_if(Predecessors.begin(),
                                        Predecessors.end(),
                                        [NewNode](auto WeakNode) {
                                          return WeakNode.lock() == NewNode;
                                        }),
                         Predecessors.end());

      // Add all Output nodes from the subgraph as predecessors for this node
      // instead
      for (auto &Output : Outputs) {
        Output->registerSuccessor(SuccNode, Output);
      }
    }

    // Remove single subgraph node and add all new individual subgraph nodes
    // to the node storage in its place
    auto OldPositionIt =
        NewNodes.erase(std::find(NewNodes.begin(), NewNodes.end(), NewNode));
    // Also set the iterator to the newly added nodes so we can continue
    // iterating over all remaining nodes
    auto InsertIt = NewNodes.insert(OldPositionIt, NewSubgraphNodes.begin(),
                                    NewSubgraphNodes.end());
    // Since the new reverse_iterator will be at i - 1 we need to advance it
    // when constructing
    NewNodeIt = std::make_reverse_iterator(std::next(InsertIt));
  }

  // Store all the new nodes locally
  MNodeStorage.insert(MNodeStorage.begin(), NewNodes.begin(), NewNodes.end());
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
  } else {
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
    }
  }

  for (uint32_t i = 0; i < MNodeStorage.size(); ++i) {
    MIDCache.insert(
        std::make_pair(GraphImpl->MNodeStorage[i]->MID, MNodeStorage[i]));
  }

  update(GraphImpl->MNodeStorage);
}

void exec_graph_impl::update(std::shared_ptr<node_impl> Node) {
  this->update(std::vector<std::shared_ptr<node_impl>>{Node});
}

void exec_graph_impl::update(
    const std::vector<std::shared_ptr<node_impl>> Nodes) {

  if (!MIsUpdatable) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "update() cannot be called on a executable graph "
                          "which was not created with property::updatable");
  }

  // If there are any accessor requirements, we have to update through the
  // scheduler to ensure that any allocations have taken place before trying to
  // update.
  bool NeedScheduledUpdate = false;
  std::vector<sycl::detail::AccessorImplHost *> UpdateRequirements;
  // At worst we may have as many requirements as there are for the entire graph
  // for updating.
  UpdateRequirements.reserve(MRequirements.size());
  for (auto &Node : Nodes) {
    // Check if node(s) derived from this modifiable node exists in this graph
    if (MIDCache.count(Node->getID()) == 0) {
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "Node passed to update() is not part of the graph.");
    }

    if (!(Node->isEmpty() || Node->MCGType == sycl::detail::CG::Kernel ||
          Node->MCGType == sycl::detail::CG::Barrier)) {
      throw sycl::exception(errc::invalid,
                            "Unsupported node type for update. Only kernel, "
                            "barrier and empty nodes are supported.");
    }

    if (const auto &CG = Node->MCommandGroup;
        CG && CG->getRequirements().size() != 0) {
      NeedScheduledUpdate = true;

      UpdateRequirements.insert(UpdateRequirements.end(),
                                Node->MCommandGroup->getRequirements().begin(),
                                Node->MCommandGroup->getRequirements().end());
    }
  }

  // Clean up any execution events which have finished so we don't pass them to
  // the scheduler.
  for (auto It = MExecutionEvents.begin(); It != MExecutionEvents.end();) {
    if ((*It)->isCompleted()) {
      It = MExecutionEvents.erase(It);
      continue;
    }
    ++It;
  }

  // If we have previous execution events do the update through the scheduler to
  // ensure it is ordered correctly.
  NeedScheduledUpdate |= MExecutionEvents.size() > 0;

  if (NeedScheduledUpdate) {
    auto AllocaQueue = std::make_shared<sycl::detail::queue_impl>(
        sycl::detail::getSyclObjImpl(MGraphImpl->getDevice()),
        sycl::detail::getSyclObjImpl(MGraphImpl->getContext()),
        sycl::async_handler{}, sycl::property_list{});
    // Don't need to care about the return event here because it is synchronous
    sycl::detail::Scheduler::getInstance().addCommandGraphUpdate(
        this, Nodes, AllocaQueue, UpdateRequirements, MExecutionEvents);
  } else {
    for (auto &Node : Nodes) {
      updateImpl(Node);
    }
  }

  // Rebuild cached requirements for this graph with updated nodes
  MRequirements.clear();
  for (auto &Node : MNodeStorage) {
    if (!Node->MCommandGroup)
      continue;
    MRequirements.insert(MRequirements.end(),
                         Node->MCommandGroup->getRequirements().begin(),
                         Node->MCommandGroup->getRequirements().end());
  }
}

void exec_graph_impl::updateImpl(std::shared_ptr<node_impl> Node) {
  // Kernel node update is the only command type supported in UR for update.
  // Updating any other types of nodes, e.g. empty & barrier nodes is a no-op.
  if (Node->MCGType != sycl::detail::CG::Kernel) {
    return;
  }
  auto ContextImpl = sycl::detail::getSyclObjImpl(MContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  auto DeviceImpl = sycl::detail::getSyclObjImpl(MGraphImpl->getDevice());

  // Gather arg information from Node
  auto &ExecCG =
      *(static_cast<sycl::detail::CGExecKernel *>(Node->MCommandGroup.get()));
  // Copy args because we may modify them
  std::vector<sycl::detail::ArgDesc> NodeArgs = ExecCG.getArguments();
  // Copy NDR desc since we need to modify it
  auto NDRDesc = ExecCG.MNDRDesc;

  pi_kernel PiKernel = nullptr;
  pi_program PiProgram = nullptr;
  auto Kernel = ExecCG.MSyclKernel;
  auto KernelBundleImplPtr = ExecCG.MKernelBundle;
  std::shared_ptr<sycl::detail::kernel_impl> SyclKernelImpl = nullptr;
  const sycl::detail::KernelArgMask *EliminatedArgMask = nullptr;

  // Use kernel_bundle if available unless it is interop.
  // Interop bundles can't be used in the first branch, because the kernels
  // in interop kernel bundles (if any) do not have kernel_id
  // and can therefore not be looked up, but since they are self-contained
  // they can simply be launched directly.
  if (KernelBundleImplPtr && !KernelBundleImplPtr->isInterop()) {
    auto KernelName = ExecCG.MKernelName;
    kernel_id KernelID =
        sycl::detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);
    kernel SyclKernel =
        KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);
    SyclKernelImpl = sycl::detail::getSyclObjImpl(SyclKernel);
    PiKernel = SyclKernelImpl->getHandleRef();
    EliminatedArgMask = SyclKernelImpl->getKernelArgMask();
  } else if (Kernel != nullptr) {
    PiKernel = Kernel->getHandleRef();
    EliminatedArgMask = Kernel->getKernelArgMask();
  } else {
    std::tie(PiKernel, std::ignore, EliminatedArgMask, PiProgram) =
        sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
            ContextImpl, DeviceImpl, ExecCG.MKernelName);
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
    Plugin->call<sycl::detail::PiApiKind::piKernelGetGroupInfo>(
        PiKernel, DeviceImpl->getHandleRef(),
        PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize,
        /* param_value_size_ret = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }
  // Create update descriptor

  // Storage for individual arg descriptors
  std::vector<pi_ext_command_buffer_update_memobj_arg_desc_t> MemobjDescs;
  std::vector<pi_ext_command_buffer_update_pointer_arg_desc_t> PtrDescs;
  std::vector<pi_ext_command_buffer_update_value_arg_desc_t> ValueDescs;
  MemobjDescs.reserve(MaskedArgs.size());
  PtrDescs.reserve(MaskedArgs.size());
  ValueDescs.reserve(MaskedArgs.size());

  pi_ext_command_buffer_update_kernel_launch_desc UpdateDesc;

  // Collect arg descriptors and fill kernel launch descriptor
  using sycl::detail::kernel_param_kind_t;
  for (size_t i = 0; i < MaskedArgs.size(); i++) {
    auto &NodeArg = MaskedArgs[i];
    switch (NodeArg.MType) {
    case kernel_param_kind_t::kind_pointer: {
      PtrDescs.push_back({static_cast<uint32_t>(NodeArg.MIndex), NodeArg.MPtr});
    } break;
    case kernel_param_kind_t::kind_std_layout: {
      ValueDescs.push_back({static_cast<uint32_t>(NodeArg.MIndex),
                            static_cast<uint32_t>(NodeArg.MSize),
                            NodeArg.MPtr});
    } break;
    case kernel_param_kind_t::kind_accessor: {
      sycl::detail::Requirement *Req =
          static_cast<sycl::detail::Requirement *>(NodeArg.MPtr);

      pi_mem_obj_property MemObjData{};

      switch (Req->MAccessMode) {
      case access::mode::read: {
        MemObjData.mem_access = PI_ACCESS_READ_ONLY;
        break;
      }
      case access::mode::write:
      case access::mode::discard_write: {
        MemObjData.mem_access = PI_ACCESS_WRITE_ONLY;
        break;
      }
      default: {
        MemObjData.mem_access = PI_ACCESS_READ_WRITE;
        break;
      }
      }
      MemObjData.type = PI_KERNEL_ARG_MEM_OBJ_ACCESS;
      MemobjDescs.push_back(pi_ext_command_buffer_update_memobj_arg_desc_t{
          static_cast<uint32_t>(NodeArg.MIndex), &MemObjData,
          static_cast<pi_mem>(Req->MData)});

    } break;

    default:
      break;
    }
  }

  UpdateDesc.num_mem_obj_args = MemobjDescs.size();
  UpdateDesc.mem_obj_arg_list = MemobjDescs.data();
  UpdateDesc.num_ptr_args = PtrDescs.size();
  UpdateDesc.ptr_arg_list = PtrDescs.data();
  UpdateDesc.num_value_args = ValueDescs.size();
  UpdateDesc.value_arg_list = ValueDescs.data();

  UpdateDesc.global_work_offset = &NDRDesc.GlobalOffset[0];
  UpdateDesc.global_work_size = &NDRDesc.GlobalSize[0];
  UpdateDesc.local_work_size = LocalSize;
  UpdateDesc.num_work_dim = NDRDesc.Dims;

  // Query the ID cache to find the equivalent exec node for the node passed to
  // this function.
  // TODO: Handle subgraphs or any other cases where multiple nodes may be
  // associated with a single key, once those node types are supported for
  // update.
  auto ExecNode = MIDCache.find(Node->MID);
  assert(ExecNode != MIDCache.end() && "Node ID was not found in ID cache");

  // Update ExecNode with new values from Node, in case we ever need to
  // rebuild the command buffers
  ExecNode->second->updateFromOtherNode(Node);

  sycl::detail::pi::PiExtCommandBufferCommand Command =
      MCommandMap[ExecNode->second];
  pi_result Res = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextCommandBufferUpdateKernelLaunch>(
      Command, &UpdateDesc);

  if (PiProgram) {
    // We retained these objects by calling getOrCreateKernel()
    Plugin->call<sycl::detail::PiApiKind::piKernelRelease>(PiKernel);
    Plugin->call<sycl::detail::PiApiKind::piProgramRelease>(PiProgram);
  }

  if (Res != PI_SUCCESS) {
    throw sycl::exception(errc::invalid, "Error updating command_graph");
  }
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

node modifiable_command_graph::addImpl(const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  graph_impl::WriteLock Lock(impl->MMutex);
  std::shared_ptr<detail::node_impl> NodeImpl = impl->add(impl, DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

node modifiable_command_graph::addImpl(std::function<void(handler &)> CGF,
                                       const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  graph_impl::WriteLock Lock(impl->MMutex);
  std::shared_ptr<detail::node_impl> NodeImpl =
      impl->add(impl, CGF, {}, DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

void modifiable_command_graph::addGraphLeafDependencies(node Node) {
  // Find all exit nodes in the current graph and add them to the dependency
  // vector
  std::shared_ptr<detail::node_impl> DstImpl =
      sycl::detail::getSyclObjImpl(Node);
  graph_impl::WriteLock Lock(impl->MMutex);
  for (auto &NodeImpl : impl->MNodeStorage) {
    if ((NodeImpl->MSuccessors.size() == 0) && (NodeImpl != DstImpl)) {
      impl->makeEdge(NodeImpl, DstImpl);
    }
  }
}

void modifiable_command_graph::make_edge(node &Src, node &Dest) {
  std::shared_ptr<detail::node_impl> SenderImpl =
      sycl::detail::getSyclObjImpl(Src);
  std::shared_ptr<detail::node_impl> ReceiverImpl =
      sycl::detail::getSyclObjImpl(Dest);

  graph_impl::WriteLock Lock(impl->MMutex);
  impl->makeEdge(SenderImpl, ReceiverImpl);
}

command_graph<graph_state::executable>
modifiable_command_graph::finalize(const sycl::property_list &PropList) const {
  // Graph is read and written in this scope so we lock
  // this graph with full priviledges.
  graph_impl::WriteLock Lock(impl->MMutex);
  return command_graph<graph_state::executable>{
      this->impl, this->impl->getContext(), PropList};
}

void modifiable_command_graph::begin_recording(
    queue &RecordingQueue, const sycl::property_list &PropList) {
  std::ignore = PropList;

  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  assert(QueueImpl);
  if (QueueImpl->get_context() != impl->getContext()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose context "
                          "differs from the graph context.");
  }
  if (QueueImpl->get_device() != impl->getDevice()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose device "
                          "differs from the graph device.");
  }

  if (QueueImpl->is_in_fusion_mode()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "SYCL queue in kernel in fusion mode "
                          "can NOT be recorded.");
  }

  if (QueueImpl->get_context() != impl->getContext()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose context "
                          "differs from the graph context.");
  }
  if (QueueImpl->get_device() != impl->getDevice()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue whose device "
                          "differs from the graph device.");
  }

  if (QueueImpl->getCommandGraph() == nullptr) {
    QueueImpl->setCommandGraph(impl);
    graph_impl::WriteLock Lock(impl->MMutex);
    impl->addQueue(QueueImpl);
  }
  if (QueueImpl->getCommandGraph() != impl) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue which is already "
                          "recording to a different graph.");
  }
}

void modifiable_command_graph::begin_recording(
    const std::vector<queue> &RecordingQueues,
    const sycl::property_list &PropList) {
  for (queue Queue : RecordingQueues) {
    this->begin_recording(Queue, PropList);
  }
}

void modifiable_command_graph::end_recording() {
  graph_impl::WriteLock Lock(impl->MMutex);
  impl->clearQueues();
}

void modifiable_command_graph::end_recording(queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl && QueueImpl->getCommandGraph() == impl) {
    QueueImpl->setCommandGraph(nullptr);
    graph_impl::WriteLock Lock(impl->MMutex);
    impl->removeQueue(QueueImpl);
  }
  if (QueueImpl->getCommandGraph() != nullptr) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "end_recording called for a queue which is recording "
                          "to a different graph.");
  }
}

void modifiable_command_graph::end_recording(
    const std::vector<queue> &RecordingQueues) {
  for (queue Queue : RecordingQueues) {
    this->end_recording(Queue);
  }
}

void modifiable_command_graph::print_graph(std::string path,
                                           bool verbose) const {
  graph_impl::ReadLock Lock(impl->MMutex);
  if (path.substr(path.find_last_of(".") + 1) == "dot") {
    impl->printGraphAsDot(path, verbose);
  } else {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "DOT graph is the only format supported at the moment.");
  }
}

std::vector<node> modifiable_command_graph::get_nodes() const {
  return createNodesFromImpls(impl->MNodeStorage);
}
std::vector<node> modifiable_command_graph::get_root_nodes() const {
  auto &Roots = impl->MRoots;
  std::vector<std::weak_ptr<node_impl>> Impls{};

  std::copy(Roots.begin(), Roots.end(), std::back_inserter(Impls));
  return createNodesFromImpls(Impls);
}

executable_command_graph::executable_command_graph(
    const std::shared_ptr<detail::graph_impl> &Graph, const sycl::context &Ctx,
    const property_list &PropList)
    : impl(std::make_shared<detail::exec_graph_impl>(Ctx, Graph, PropList)) {
  finalizeImpl(); // Create backend representation for executable graph
}

void executable_command_graph::finalizeImpl() {
  impl->makePartitions();

  auto Device = impl->getGraphImpl()->getDevice();
  for (auto Partition : impl->getPartitions()) {
    if (!Partition->isHostTask()) {
      impl->createCommandBuffers(Device, Partition);
    }
  }
}

void executable_command_graph::update(
    const command_graph<graph_state::modifiable> &Graph) {
  impl->update(sycl::detail::getSyclObjImpl(Graph));
}

void executable_command_graph::update(const node &Node) {
  impl->update(sycl::detail::getSyclObjImpl(Node));
}

void executable_command_graph::update(const std::vector<node> &Nodes) {
  std::vector<std::shared_ptr<node_impl>> NodeImpls{};
  NodeImpls.reserve(Nodes.size());
  for (auto &Node : Nodes) {
    NodeImpls.push_back(sycl::detail::getSyclObjImpl(Node));
  }

  impl->update(NodeImpls);
}

dynamic_parameter_base::dynamic_parameter_base(
    command_graph<graph_state::modifiable> Graph, size_t ParamSize,
    const void *Data)
    : impl(std::make_shared<dynamic_parameter_impl>(
          sycl::detail::getSyclObjImpl(Graph), ParamSize, Data)) {}

void dynamic_parameter_base::updateValue(const void *NewValue, size_t Size) {
  impl->updateValue(NewValue, Size);
}

void dynamic_parameter_base::updateAccessor(
    const sycl::detail::AccessorBaseHost *Acc) {
  impl->updateAccessor(Acc);
}

} // namespace detail

node_type node::get_type() const { return impl->MNodeType; }

std::vector<node> node::get_predecessors() const {
  return detail::createNodesFromImpls(impl->MPredecessors);
}

std::vector<node> node::get_successors() const {
  return detail::createNodesFromImpls(impl->MSuccessors);
}

node node::get_node_from_event(event nodeEvent) {
  auto EventImpl = sycl::detail::getSyclObjImpl(nodeEvent);
  auto GraphImpl = EventImpl->getCommandGraph();

  return sycl::detail::createSyclObjFromImpl<node>(
      GraphImpl->getNodeForEvent(EventImpl));
}

template <> __SYCL_EXPORT void node::update_nd_range<1>(nd_range<1> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_nd_range<2>(nd_range<2> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_nd_range<3>(nd_range<3> NDRange) {
  impl->updateNDRange(NDRange);
}
template <> __SYCL_EXPORT void node::update_range<1>(range<1> Range) {
  impl->updateRange(Range);
}
template <> __SYCL_EXPORT void node::update_range<2>(range<2> Range) {
  impl->updateRange(Range);
}
template <> __SYCL_EXPORT void node::update_range<3>(range<3> Range) {
  impl->updateRange(Range);
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
