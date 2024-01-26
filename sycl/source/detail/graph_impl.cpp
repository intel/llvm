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

// Developer switch to use emulation mode on all backends, even those that
// report native support, this is useful for debugging.
#define FORCE_EMULATION_MODE 0

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

void duplicateNode(const std::shared_ptr<node_impl> Node,
                   std::shared_ptr<node_impl> &NodeCopy) {
  if (Node->MCGType == sycl::detail::CG::None) {
    NodeCopy = std::make_shared<node_impl>();
    NodeCopy->MCGType = sycl::detail::CG::None;
  } else {
    NodeCopy = std::make_shared<node_impl>(Node->MCGType, Node->getCGCopy());
  }
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
  for (auto Predecessor : Node->MPredecessors) {
    if (Predecessor.lock()->MPartitionNum == Node->MPartitionNum) {
      return false;
    }
  }
  return true;
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
  for (auto Node : MGraphImpl->MNodeStorage) {
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
      for (auto HT : HostTaskList) {
        auto HTPartitionNum = HT->MPartitionNum;
        if (HTPartitionNum != -1) {
          // can merge predecessors of node `Node` with predecessors of node
          // `HT` (HTPartitionNum-1) since HT must be reprocessed
          for (auto NodeImpl : MGraphImpl->MNodeStorage) {
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
    for (auto Node : MGraphImpl->MNodeStorage) {
      if (Node->MPartitionNum == i) {
        MPartitionNodes[Node] = PartitionFinalNum;
        if (isPartitionRoot(Node)) {
          Partition->MRoots.insert(Node);
        }
      }
    }
    if (Partition->MRoots.size() > 0) {
      Partition->schedule();
      MPartitions.push_back(Partition);
      PartitionFinalNum++;
    }
  }

  // Add an empty partition if there is no partition, i.e. empty graph
  if (MPartitions.size() == 0) {
    MPartitions.push_back(std::make_shared<partition>());
  }

  // Make global schedule list
  for (auto Partition : MPartitions) {
    MSchedule.insert(MSchedule.end(), Partition->MSchedule.begin(),
                     Partition->MSchedule.end());
  }

  // Reset node groups (if node have to be re-processed - e.g. subgraph)
  for (auto &Node : MGraphImpl->MNodeStorage) {
    Node->MPartitionNum = -1;
  }
}

graph_impl::~graph_impl() {
  clearQueues();
  for (auto &MemObj : MMemObjs) {
    MemObj->markNoLongerBeingUsedInGraph();
  }
}

std::shared_ptr<node_impl> graph_impl::addNodesToExits(
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
  }

  return this->add(Outputs);
}

std::shared_ptr<node_impl> graph_impl::addSubgraphNodes(
    const std::shared_ptr<exec_graph_impl> &SubGraphExec) {
  std::map<std::shared_ptr<node_impl>, std::shared_ptr<node_impl>> NodesMap;

  std::list<std::shared_ptr<node_impl>> NodesList = SubGraphExec->getSchedule();
  std::list<std::shared_ptr<node_impl>> NewNodesList{NodesList.size()};

  // Duplication of nodes
  for (auto NodeIt = NodesList.end(), NewNodesIt = NewNodesList.end();
       NodeIt != NodesList.begin();) {
    --NodeIt;
    --NewNodesIt;
    auto Node = *NodeIt;
    std::shared_ptr<node_impl> NodeCopy;
    duplicateNode(Node, NodeCopy);
    *NewNodesIt = NodeCopy;
    NodesMap.insert({Node, NodeCopy});
    for (auto &NextNode : Node->MSuccessors) {
      auto Successor = NodesMap.at(NextNode.lock());
      NodeCopy->registerSuccessor(Successor, NodeCopy);
    }
  }

  return addNodesToExits(NewNodesList);
}

void graph_impl::addRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.insert(Root);
}

void graph_impl::removeRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.erase(Root);
}

std::shared_ptr<node_impl>
graph_impl::add(const std::vector<std::shared_ptr<node_impl>> &Dep) {
  // Copy deps so we can modify them
  auto Deps = Dep;

  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>();

  // Add any deps from the vector of extra dependencies
  Deps.insert(Deps.end(), MExtraDependencies.begin(), MExtraDependencies.end());

  MNodeStorage.push_back(NodeImpl);

  addDepsToNode(NodeImpl, Deps);

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
  Handler.finalize();

  if (Handler.MCGType == sycl::detail::CG::Barrier) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "The sycl_ext_oneapi_enqueue_barrier feature is not available with "
        "SYCL Graph Explicit API. Please use empty nodes instead.");
  }

  // If the handler recorded a subgraph return that here as the relevant nodes
  // have already been added. The node returned here is an empty node with
  // dependencies on all the exit nodes of the subgraph.
  if (Handler.MSubgraphNode) {
    return Handler.MSubgraphNode;
  }
  return this->add(Handler.MCGType, std::move(Handler.MGraphNodeCG), Dep);
}

std::shared_ptr<node_impl>
graph_impl::add(const std::vector<sycl::detail::EventImplPtr> Events) {

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

  return this->add(Deps);
}

std::shared_ptr<node_impl>
graph_impl::add(sycl::detail::CG::CGTYPE CGType,
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
      if (Node->hasRequirement(Req)) {
        bool ShouldAddDep = true;
        // If any of this node's successors have this requirement then we skip
        // adding the current node as a dependency.
        for (auto &Succ : Node->MSuccessors) {
          if (Succ.lock()->hasRequirement(Req)) {
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

  // Add any deps from the extra dependencies vector
  Deps.insert(Deps.end(), MExtraDependencies.begin(), MExtraDependencies.end());

  const std::shared_ptr<node_impl> &NodeImpl =
      std::make_shared<node_impl>(CGType, std::move(CommandGroup));
  MNodeStorage.push_back(NodeImpl);

  addDepsToNode(NodeImpl, Deps);

  // Set barrier nodes as prerequisites (new start points) for subsequent nodes
  if (CGType == sycl::detail::CG::Barrier) {
    MExtraDependencies.push_back(NodeImpl);
  }

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

std::vector<sycl::detail::EventImplPtr> graph_impl::getExitNodesEvents() {
  std::vector<sycl::detail::EventImplPtr> Events;

  for (auto Node : MNodeStorage) {
    if (Node->MSuccessors.empty()) {
      Events.push_back(getEventForNode(Node));
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
  pi_int32 Res = sycl::detail::enqueueImpCommandBufferKernel(
      Ctx, DeviceImpl, CommandBuffer,
      *static_cast<sycl::detail::CGExecKernel *>((Node->MCommandGroup.get())),
      Deps, &NewSyncPoint, nullptr);

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
          Node->getCGCopy(), AllocaQueue, CommandBuffer, Deps);

  return Event->getSyncPoint();
}
void exec_graph_impl::createCommandBuffers(
    sycl::device Device, std::shared_ptr<partition> &Partition) {
  sycl::detail::pi::PiExtCommandBuffer OutCommandBuffer;
  sycl::detail::pi::PiExtCommandBufferDesc Desc{};
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

exec_graph_impl::~exec_graph_impl() {
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
}

sycl::event
exec_graph_impl::enqueue(const std::shared_ptr<sycl::detail::queue_impl> &Queue,
                         sycl::detail::CG::StorageInitHelper CGData) {
  WriteLock Lock(MMutex);

  std::vector<sycl::detail::EventImplPtr> PartitionEvents;

  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });

  sycl::detail::EventImplPtr NewEvent;
  for (auto CurrentPartition : MPartitions) {
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
            CGData.MEvents.push_back(Event);
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
                CommandBuffer, std::move(CGData));

        NewEvent = sycl::detail::Scheduler::getInstance().addCG(
            std::move(CommandGroup), Queue);
      }
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
          NodeImpl->getCGCopy(), Queue);
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
              PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT);
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
                  NodeImpl->getCGCopy(), Queue);

          ScheduledEvents.push_back(EventImpl);
        }
      }
      // Create an event which has all kernel events as dependencies
      NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
      NewEvent->setStateIncomplete();
      NewEvent->getPreparedDepsEvents() = ScheduledEvents;
    }
    NewEvent->setEventFromSubmittedExecGraph(true);
    CGData.MEvents.push_back(NewEvent);
  }

  // Keep track of this execution event so we can make sure it's completed in
  // the destructor.
  MExecutionEvents.push_back(NewEvent);
  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  return QueueEvent;
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
  std::shared_ptr<detail::node_impl> NodeImpl = impl->add(DepImpls);
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
modifiable_command_graph::finalize(const sycl::property_list &) const {
  // Graph is read and written in this scope so we lock
  // this graph with full priviledges.
  graph_impl::WriteLock Lock(impl->MMutex);
  return command_graph<graph_state::executable>{this->impl,
                                                this->impl->getContext()};
}

bool modifiable_command_graph::begin_recording(queue &RecordingQueue) {
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
    return true;
  }
  if (QueueImpl->getCommandGraph() != impl) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "begin_recording called for a queue which is already "
                          "recording to a different graph.");
  }
  // Queue was already recording to this graph.
  return false;
}

bool modifiable_command_graph::begin_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (queue Queue : RecordingQueues) {
    QueueStateChanged |= this->begin_recording(Queue);
  }
  return QueueStateChanged;
}

bool modifiable_command_graph::end_recording() {
  graph_impl::WriteLock Lock(impl->MMutex);
  return impl->clearQueues();
}

bool modifiable_command_graph::end_recording(queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl && QueueImpl->getCommandGraph() == impl) {
    QueueImpl->setCommandGraph(nullptr);
    graph_impl::WriteLock Lock(impl->MMutex);
    impl->removeQueue(QueueImpl);
    return true;
  }
  if (QueueImpl->getCommandGraph() != nullptr) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "end_recording called for a queue which is recording "
                          "to a different graph.");
  }

  // Queue was not recording to a graph.
  return false;
}

bool modifiable_command_graph::end_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (queue Queue : RecordingQueues) {
    QueueStateChanged |= this->end_recording(Queue);
  }
  return QueueStateChanged;
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

executable_command_graph::executable_command_graph(
    const std::shared_ptr<detail::graph_impl> &Graph, const sycl::context &Ctx)
    : impl(std::make_shared<detail::exec_graph_impl>(Ctx, Graph)) {
  finalizeImpl(); // Create backend representation for executable graph
}

void executable_command_graph::finalizeImpl() {
  impl->makePartitions();

  auto Device = impl->getGraphImpl()->getDevice();
  bool CmdBufSupport =
      Device
          .get_info<ext::oneapi::experimental::info::device::graph_support>() ==
      graph_support_level::native;

#if FORCE_EMULATION_MODE
  // Above query should still succeed in emulation mode, but ignore the
  // result and use emulation.
  CmdBufSupport = false;
#endif
  if (CmdBufSupport) {
    for (auto Partition : impl->getPartitions()) {
      if (!Partition->isHostTask()) {
        impl->createCommandBuffers(Device, Partition);
      }
    }
  }
}

void executable_command_graph::update(
    const command_graph<graph_state::modifiable> &Graph) {
  (void)Graph;
  throw sycl::exception(sycl::make_error_code(errc::invalid),
                        "Method not yet implemented");
}

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
