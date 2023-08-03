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
#include <sycl/feature_test.hpp>
#include <sycl/queue.hpp>

#include <deque>

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

/// Recursively check if a given node is an exit node, and add the new nodes as
/// successors if so.
/// @param[in] CurrentNode Node to check as exit node.
/// @param[in] NewInputs Noes to add as successors.
void connectToExitNodes(
    std::shared_ptr<node_impl> CurrentNode,
    const std::vector<std::shared_ptr<node_impl>> &NewInputs) {
  if (CurrentNode->MSuccessors.size() > 0) {
    for (auto Successor : CurrentNode->MSuccessors) {
      connectToExitNodes(Successor, NewInputs);
    }

  } else {
    for (auto Input : NewInputs) {
      CurrentNode->registerSuccessor(Input, CurrentNode);
    }
  }
}

/// Recursive check if a graph node or its successors contains a given
/// requirement.
/// @param[in] Req The requirement to check for.
/// @param[in] CurrentNode The current graph node being checked.
/// @param[in,out] Deps The unique list of dependencies which have been
/// identified for this requirement.
/// @return True if a dependency was added in this node or any of its
/// successors.
bool checkForRequirement(sycl::detail::AccessorImplHost *Req,
                         const std::shared_ptr<node_impl> &CurrentNode,
                         std::set<std::shared_ptr<node_impl>> &Deps) {
  bool SuccessorAddedDep = false;
  for (auto &Successor : CurrentNode->MSuccessors) {
    SuccessorAddedDep |= checkForRequirement(Req, Successor, Deps);
  }

  if (!CurrentNode->isEmpty() && Deps.find(CurrentNode) == Deps.end() &&
      CurrentNode->hasRequirement(Req) && !SuccessorAddedDep) {
    Deps.insert(CurrentNode);
    return true;
  }
  return SuccessorAddedDep;
}

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
    if (visitNodeDepthFirst(Successor, VisitedNodes, NodeStack, NodeFunc)) {
      return true;
    }
  }
  NodeStack.pop_back();
  return false;
}
} // anonymous namespace

void exec_graph_impl::schedule() {
  if (MSchedule.empty()) {
    for (auto Node : MGraphImpl->MRoots) {
      Node->sortTopological(Node, MSchedule);
    }
  }
}

std::shared_ptr<node_impl> graph_impl::addSubgraphNodes(
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

  // Recursively walk the graph to find exit nodes and connect up the inputs
  // TODO: Consider caching exit nodes so we don't have to do this
  for (auto NodeImpl : MRoots) {
    connectToExitNodes(NodeImpl, Inputs);
  }

  return this->add(Outputs);
}

void graph_impl::addRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.insert(Root);
}

void graph_impl::removeRoot(const std::shared_ptr<node_impl> &Root) {
  MRoots.erase(Root);
}

std::shared_ptr<node_impl>
graph_impl::add(const std::vector<std::shared_ptr<node_impl>> &Dep) {
  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>();

  // TODO: Encapsulate in separate function to avoid duplication
  if (!Dep.empty()) {
    for (auto N : Dep) {
      N->registerSuccessor(NodeImpl, N); // register successor
      this->removeRoot(NodeImpl);        // remove receiver from root node
                                         // list
    }
  } else {
    this->addRoot(NodeImpl);
  }

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
  for (auto Dep : Events) {
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
  for (auto &Req : Requirements) {
    // Look through the graph for nodes which share this requirement
    for (auto NodePtr : MRoots) {
      checkForRequirement(Req, NodePtr, UniqueDeps);
    }
  }

  // Add any nodes specified by event dependencies into the dependency list
  for (auto Dep : CommandGroup->getEvents()) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      if (UniqueDeps.find(NodeImpl->second) == UniqueDeps.end()) {
        UniqueDeps.insert(NodeImpl->second);
      }
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
      std::make_shared<node_impl>(CGType, std::move(CommandGroup));
  if (!Deps.empty()) {
    for (auto N : Deps) {
      N->registerSuccessor(NodeImpl, N); // register successor
      this->removeRoot(NodeImpl);        // remove receiver from root node
                                         // list
    }
  } else {
    this->addRoot(NodeImpl);
  }
  return NodeImpl;
}

bool graph_impl::clearQueues() {
  bool AnyQueuesCleared = false;
  for (auto &Queue : MRecordingQueues) {
    Queue->setCommandGraph(nullptr);
    AnyQueuesCleared = true;
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
    if (visitNodeDepthFirst(Root, VisitedNodes, NodeStack, NodeFunc)) {
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
  auto CheckForNodes = [&](std::shared_ptr<node_impl> &Node,
                           std::deque<std::shared_ptr<node_impl>> &) {
    if (Node == Src) {
      SrcFound = true;
    }
    if (Node == Dest) {
      DestFound = true;
    }
    return SrcFound && DestFound;
  };

  searchDepthFirst(CheckForNodes);

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

// Check if nodes are empty and if so loop back through predecessors until we
// find the real dependency.
void exec_graph_impl::findRealDeps(
    std::vector<sycl::detail::pi::PiExtSyncPoint> &Deps,
    std::shared_ptr<node_impl> CurrentNode) {
  if (CurrentNode->isEmpty()) {
    for (auto &N : CurrentNode->MPredecessors) {
      auto NodeImpl = N.lock();
      findRealDeps(Deps, NodeImpl);
    }
  } else {
    // Verify that the sync point has actually been set for this node.
    auto SyncPoint = MPiSyncPoints.find(CurrentNode);
    assert(SyncPoint != MPiSyncPoints.end() &&
           "No sync point has been set for node dependency.");
    // Check if the dependency has already been added.
    if (std::find(Deps.begin(), Deps.end(), SyncPoint->second) == Deps.end()) {
      Deps.push_back(SyncPoint->second);
    }
  }
}

sycl::detail::pi::PiExtSyncPoint exec_graph_impl::enqueueNodeDirect(
    sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    std::shared_ptr<node_impl> Node) {
  std::vector<sycl::detail::pi::PiExtSyncPoint> Deps;
  for (auto &N : Node->MPredecessors) {
    findRealDeps(Deps, N.lock());
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
    findRealDeps(Deps, N.lock());
  }

  sycl::detail::EventImplPtr Event =
      sycl::detail::Scheduler::getInstance().addCG(
          Node->getCGCopy(), AllocaQueue, CommandBuffer, Deps);

  return Event->getSyncPoint();
}
void exec_graph_impl::createCommandBuffers(sycl::device Device) {
  // TODO we only have a single command-buffer per graph here, but
  // this will need to be multiple command-buffers for non-trivial graphs
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

  MPiCommandBuffers[Device] = OutCommandBuffer;

  // TODO extract kernel bundle logic from enqueueImpKernel
  for (auto Node : MSchedule) {
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
  // clear all recording queue if not done before (no call to end_recording)
  MGraphImpl->clearQueues();

  const sycl::detail::PluginPtr &Plugin =
      sycl::detail::getSyclObjImpl(MContext)->getPlugin();
  MSchedule.clear();
  // We need to wait on all command buffer executions before we can release
  // them.
  for (auto &Event : MExecutionEvents) {
    Event->wait(Event);
  }

  for (auto Iter : MPiCommandBuffers) {
    if (auto CmdBuf = Iter.second; CmdBuf) {
      pi_result Res = Plugin->call_nocheck<
          sycl::detail::PiApiKind::piextCommandBufferRelease>(CmdBuf);
      (void)Res;
      assert(Res == pi_result::PI_SUCCESS);
    }
  }
}

sycl::event
exec_graph_impl::enqueue(const std::shared_ptr<sycl::detail::queue_impl> &Queue,
                         sycl::detail::CG::StorageInitHelper CGData) {
  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });

  auto CommandBuffer = MPiCommandBuffers[Queue->get_device()];
  sycl::detail::EventImplPtr NewEvent;

  if (CommandBuffer) {
    NewEvent = CreateNewEvent();
    sycl::detail::pi::PiEvent *OutEvent = &NewEvent->getHandleRef();
    // Merge requirements from the nodes into requirements (if any) from the
    // handler.
    CGData.MRequirements.insert(CGData.MRequirements.end(),
                                MRequirements.begin(), MRequirements.end());
    // If we have no requirements or dependent events for the command buffer,
    // enqueue it directly
    if (CGData.MRequirements.empty() && CGData.MEvents.empty()) {
      pi_result Res =
          Queue->getPlugin()
              ->call_nocheck<
                  sycl::detail::PiApiKind::piextEnqueueCommandBuffer>(
                  CommandBuffer, Queue->getHandleRef(), 0, nullptr, OutEvent);
      if (Res != pi_result::PI_SUCCESS) {
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
  } else {
    std::vector<std::shared_ptr<sycl::detail::event_impl>> ScheduledEvents;
    for (auto &NodeImpl : MSchedule) {
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
        NewEvent = CreateNewEvent();
        sycl::detail::pi::PiEvent *OutEvent = &NewEvent->getHandleRef();
        pi_int32 Res = sycl::detail::enqueueImpKernel(
            Queue, CG->MNDRDesc, CG->MArgs,
            // TODO: Handler KernelBundles
            nullptr, CG->MSyclKernel, CG->MKernelName, RawEvents, OutEvent,
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
      } else {

        sycl::detail::EventImplPtr EventImpl =
            sycl::detail::Scheduler::getInstance().addCG(NodeImpl->getCGCopy(),
                                                         Queue);

        ScheduledEvents.push_back(EventImpl);
      }
    }
    // Create an event which has all kernel events as dependencies
    NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setStateIncomplete();
    NewEvent->getPreparedDepsEvents() = ScheduledEvents;
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

node modifiable_command_graph::addImpl(const std::vector<node> &Deps) {
  impl->throwIfGraphRecordingQueue("Explicit API \"Add()\" function");
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

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

  std::shared_ptr<detail::node_impl> NodeImpl =
      impl->add(impl, CGF, {}, DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

void modifiable_command_graph::make_edge(node &Src, node &Dest) {
  std::shared_ptr<detail::node_impl> SenderImpl =
      sycl::detail::getSyclObjImpl(Src);
  std::shared_ptr<detail::node_impl> ReceiverImpl =
      sycl::detail::getSyclObjImpl(Dest);

  impl->makeEdge(SenderImpl, ReceiverImpl);
}

command_graph<graph_state::executable>
modifiable_command_graph::finalize(const sycl::property_list &) const {
  return command_graph<graph_state::executable>{this->impl,
                                                this->impl->getContext()};
}

bool modifiable_command_graph::begin_recording(queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);

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

bool modifiable_command_graph::end_recording() { return impl->clearQueues(); }

bool modifiable_command_graph::end_recording(queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl->getCommandGraph() == impl) {
    QueueImpl->setCommandGraph(nullptr);
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

executable_command_graph::executable_command_graph(
    const std::shared_ptr<detail::graph_impl> &Graph, const sycl::context &Ctx)
    : MTag(rand()),
      impl(std::make_shared<detail::exec_graph_impl>(Ctx, Graph)) {
  finalizeImpl(); // Create backend representation for executable graph
}

void executable_command_graph::finalizeImpl() {
  // Create PI command-buffers for each device in the finalized context
  impl->schedule();

  auto Context = impl->getContext();
  for (auto Device : Context.get_devices()) {
    bool CmdBufSupport =
        Device.get_info<
            ext::oneapi::experimental::info::device::graph_support>() ==
        info::graph_support_level::native;

#if FORCE_EMULATION_MODE
    // Above query should still succeed in emulation mode, but ignore the
    // result and use emulation.
    CmdBufSupport = false;
#endif

    if (CmdBufSupport) {
      impl->createCommandBuffers(Device);
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
