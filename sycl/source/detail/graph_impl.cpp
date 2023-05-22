//==--------- graph_impl.cpp - SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/graph_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/handler_impl.hpp> 
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/queue.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

namespace {

/// Recursively check if a given node is an exit node, and add the new nodes as
/// successors if so.
/// @param[in] CurrentNode Node to check as exit node.
/// @param[in] NewInputs Noes to add as successors.
void connect_to_exit_nodes(
    std::shared_ptr<node_impl> CurrentNode,
    const std::vector<std::shared_ptr<node_impl>> &NewInputs) {
  if (CurrentNode->MSuccessors.size() > 0) {
    for (auto Successor : CurrentNode->MSuccessors) {
      connect_to_exit_nodes(Successor, NewInputs);
    }

  } else {
    for (auto Input : NewInputs) {
      CurrentNode->register_successor(Input, CurrentNode);
    }
  }
}

/// Recursive check if a graph node or its successors contains a given kernel
/// argument.
/// @param[in] Arg The kernel argument to check for.
/// @param[in] CurrentNode The current graph node being checked.
/// @param[in,out] Deps The unique list of dependencies which have been
/// identified for this arg.
/// @return True if a dependency was added in this node or any of its
/// successors.
bool check_for_arg(const sycl::detail::ArgDesc &Arg,
                   const std::shared_ptr<node_impl> &CurrentNode,
                   std::set<std::shared_ptr<node_impl>> &Deps) {
  bool SuccessorAddedDep = false;
  for (auto &Successor : CurrentNode->MSuccessors) {
    SuccessorAddedDep |= check_for_arg(Arg, Successor, Deps);
  }

  if (!CurrentNode->is_empty() && Deps.find(CurrentNode) == Deps.end() &&
      CurrentNode->has_arg(Arg) && !SuccessorAddedDep) {
    Deps.insert(CurrentNode);
    return true;
  }
  return SuccessorAddedDep;
}
} // anonymous namespace

void exec_graph_impl::schedule() {
  if (MSchedule.empty()) {
    for (auto Node : MGraphImpl->MRoots) {
      Node->topology_sort(Node, MSchedule);
    }
  }
}

std::shared_ptr<node_impl> graph_impl::add_subgraph_nodes(
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
    connect_to_exit_nodes(NodeImpl, Inputs);
  }

  return this->add(Outputs);
}

sycl::event
exec_graph_impl::exec(const std::shared_ptr<sycl::detail::queue_impl> &Queue) {
  sycl::event RetEvent = enqueue(Queue);
  // TODO: Remove this queue wait. Currently waiting on the event returned from
  // graph execution does not work.
  Queue->wait();

  return RetEvent;
}

void graph_impl::add_root(const std::shared_ptr<node_impl> &Root) {
  MRoots.insert(Root);
}

void graph_impl::remove_root(const std::shared_ptr<node_impl> &Root) {
  MRoots.erase(Root);
}

std::shared_ptr<node_impl>
graph_impl::add(const std::vector<std::shared_ptr<node_impl>> &Dep) {
  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>();

  // TODO: Encapsulate in separate function to avoid duplication
  if (!Dep.empty()) {
    for (auto N : Dep) {
      N->register_successor(NodeImpl, N); // register successor
      this->remove_root(NodeImpl);     // remove receiver from root node
                                       // list
    }
  } else {
    this->add_root(NodeImpl);
  }

  return NodeImpl;
}

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &Impl,
                std::function<void(handler &)> CGF,
                const std::vector<sycl::detail::ArgDesc> &Args,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
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
graph_impl::add(sycl::detail::CG::CGTYPE CGType,
                std::unique_ptr<sycl::detail::CG> CommandGroup,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
  // Copy deps so we can modify them
  auto Deps = Dep;
  if (CGType == sycl::detail::CG::Kernel) {
    // A unique set of dependencies obtained by checking kernel arguments
    // for accessors
    std::set<std::shared_ptr<node_impl>> UniqueDeps;
    const auto &Args =
        static_cast<sycl::detail::CGExecKernel *>(CommandGroup.get())->MArgs;
    for (auto &Arg : Args) {
      if (Arg.MType != sycl::detail::kernel_param_kind_t::kind_accessor) {
        continue;
      }
      // Look through the graph for nodes which share this argument
      for (auto NodePtr : MRoots) {
        check_for_arg(Arg, NodePtr, UniqueDeps);
      }
    }

    // Add any deps determined from accessor arguments into the dependency list
    Deps.insert(Deps.end(), UniqueDeps.begin(), UniqueDeps.end());
  }

  // Add any nodes specified by event dependencies into the dependency list
  for (auto Dep : CommandGroup->MEvents) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      Deps.push_back(NodeImpl->second);
    } else {
      throw sycl::exception(errc::invalid,
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    }
  }

  const std::shared_ptr<node_impl> &NodeImpl =
      std::make_shared<node_impl>(CGType, std::move(CommandGroup));
  if (!Deps.empty()) {
    for (auto N : Deps) {
      N->register_successor(NodeImpl, N); // register successor
      this->remove_root(NodeImpl);     // remove receiver from root node
                                       // list
    }
  } else {
    this->add_root(NodeImpl);
  }
  return NodeImpl;
}

bool graph_impl::clear_queues() {
  bool AnyQueuesCleared = false;
  for (auto &Queue : MRecordingQueues) {
    Queue->setCommandGraph(nullptr);
    AnyQueuesCleared = true;
  }
  MRecordingQueues.clear();

  return AnyQueuesCleared;
}

// Check if nodes are empty and if so loop back through predecessors until we
// find the real dependency.
void exec_graph_impl::find_real_deps(std::vector<RT::PiExtSyncPoint> &Deps,
                                     std::shared_ptr<node_impl> CurrentNode) {
  if (CurrentNode->is_empty()) {
    for (auto &N : CurrentNode->MPredecessors) {
      auto NodeImpl = N.lock();
      find_real_deps(Deps, NodeImpl);
    }
  } else {
    // Verify that the sync point has actually been set for this node.
    if (auto SyncPoint = MPiSyncPoints.find(CurrentNode);
        SyncPoint != MPiSyncPoints.end()) {
      // Check if the dependency has already been added.
      if (std::find(Deps.begin(), Deps.end(), SyncPoint->second) ==
          Deps.end()) {
        Deps.push_back(SyncPoint->second);
      }
    } else {
      assert(false && "No sync point has been set for node dependency.");
    }
  }
}

RT::PiExtSyncPoint exec_graph_impl::enqueue_node_direct(
    sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
    RT::PiExtCommandBuffer CommandBuffer, std::shared_ptr<node_impl> Node) {
  std::vector<RT::PiExtSyncPoint> Deps;
  for (auto &N : Node->MPredecessors) {
    find_real_deps(Deps, N.lock());
  }
  RT::PiExtSyncPoint NewSyncPoint;
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

RT::PiExtSyncPoint exec_graph_impl::enqueue_node(
    sycl::context Ctx, std::shared_ptr<sycl::detail::device_impl> DeviceImpl,
    RT::PiExtCommandBuffer CommandBuffer, std::shared_ptr<node_impl> Node) {

  // Queue which will be used for allocation operations for accessors.
  auto AllocaQueue = std::make_shared<sycl::detail::queue_impl>(
      DeviceImpl, sycl::detail::getSyclObjImpl(Ctx), sycl::async_handler{},
      sycl::property_list{});

  std::vector<RT::PiExtSyncPoint> Deps;
  for (auto &N : Node->MPredecessors) {
    find_real_deps(Deps, N.lock());
  }

  sycl::detail::EventImplPtr Event =
      sycl::detail::Scheduler::getInstance().addCG(
          std::move(Node->getCGCopy()), AllocaQueue, CommandBuffer, Deps);

  return Event->getSyncPoint();
}
void exec_graph_impl::create_pi_command_buffers(sycl::device D) {
  // TODO we only have a single command-buffer per graph here, but
  // this will need to be multiple command-buffers for non-trivial graphs
  RT::PiExtCommandBuffer OutCommandBuffer;
  RT::PiExtCommandBufferDesc Desc{};
  auto ContextImpl = sycl::detail::getSyclObjImpl(MContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  auto DeviceImpl = sycl::detail::getSyclObjImpl(D);
  pi_result Res =
      Plugin.call_nocheck<sycl::detail::PiApiKind::piextCommandBufferCreate>(
          ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(), &Desc,
          &OutCommandBuffer);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::invalid, "Failed to create PI command-buffer");
  }

  MPiCommandBuffers[D] = OutCommandBuffer;

  // TODO extract kernel bundle logic from enqueueImpKernel
  for (auto Node : MSchedule) {
    sycl::detail::CG::CGTYPE type = Node->MCGType;
    // If the node is a kernel with no special requirements we can enqueue it
    // directly.
    if (type == sycl::detail::CG::Kernel &&
        Node->MCommandGroup->MRequirements.size() +
                static_cast<sycl::detail::CGExecKernel *>(
                    Node->MCommandGroup.get())
                    ->MStreams.size() ==
            0) {
      MPiSyncPoints[Node] =
          enqueue_node_direct(MContext, DeviceImpl, OutCommandBuffer, Node);
    } else {
      MPiSyncPoints[Node] =
          enqueue_node(MContext, DeviceImpl, OutCommandBuffer, Node);
    }

    // Append Node requirements to overall graph requirements
    MRequirements.insert(MRequirements.end(),
                         Node->MCommandGroup->MRequirements.begin(),
                         Node->MCommandGroup->MRequirements.end());
  }

  Res =
      Plugin.call_nocheck<sycl::detail::PiApiKind::piextCommandBufferFinalize>(
          OutCommandBuffer);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::invalid,
                          "Failed to finalize PI command-buffer");
  }
}

exec_graph_impl::~exec_graph_impl() {
  MSchedule.clear();
  for (auto Iter : MPiCommandBuffers) {
    const sycl::detail::plugin &Plugin =
        sycl::detail::getSyclObjImpl(MContext)->getPlugin();
    auto CmdBuf = Iter.second;
    pi_result Res =
        Plugin.call_nocheck<sycl::detail::PiApiKind::piextCommandBufferRelease>(
            CmdBuf);
    (void)Res;
    assert(Res == pi_result::PI_SUCCESS);
  }
}

sycl::event exec_graph_impl::enqueue(
    const std::shared_ptr<sycl::detail::queue_impl> &Queue) {
  std::vector<RT::PiEvent> RawEvents;
  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });
#if SYCL_EXT_ONEAPI_GRAPH
  auto NewEvent = CreateNewEvent();
  RT::PiEvent *OutEvent = &NewEvent->getHandleRef();
  auto CommandBuffer = MPiCommandBuffers[Queue->get_device()];

  // If we have no requirements for accessors for the command buffer, enqueue it
  // directly
  if (MRequirements.empty()) {
    pi_result Res =
        Queue->getPlugin()
            .call_nocheck<sycl::detail::PiApiKind::piextEnqueueCommandBuffer>(
                CommandBuffer, Queue->getHandleRef(), RawEvents.size(),
                RawEvents.empty() ? nullptr : &RawEvents[0], OutEvent);
    if (Res != pi_result::PI_SUCCESS) {
      throw sycl::exception(
          errc::event, "Failed to enqueue event for command buffer submission");
    }
  } else {
    std::unique_ptr<sycl::detail::CG> CommandGroup =
        std::make_unique<sycl::detail::CGExecCommandBuffer>(CommandBuffer,
                                                            MRequirements);

    NewEvent = sycl::detail::Scheduler::getInstance().addCG(
        std::move(CommandGroup), Queue);
  }

#else
  std::vector<std::shared_ptr<sycl::detail::event_impl>> ScheduledEvents;
  for (auto &NodeImpl : MSchedule) {
    std::vector<RT::PiEvent> RawEvents;

    // If the node has no requirements for accessors etc. then we skip the
    // scheduler and enqueue directly.
    if (NodeImpl->MCGType == sycl::detail::CG::Kernel &&
        NodeImpl->MCommandGroup->MRequirements.size() +
                static_cast<sycl::detail::CGExecKernel *>(
                    NodeImpl->MCommandGroup.get())
                    ->MStreams.size() ==
            0) {
      sycl::detail::CGExecKernel *CG =
          static_cast<sycl::detail::CGExecKernel *>(
              NodeImpl->MCommandGroup.get());
      auto NewEvent = CreateNewEvent();
      RT::PiEvent *OutEvent = &NewEvent->getHandleRef();
      pi_int32 Res =
          sycl::
              detail::enqueueImpKernel(Queue, CG->MNDRDesc, CG->MArgs,
                                       nullptr /* TODO: Handle KernelBundles */,
                                       CG->MSyclKernel, CG->MKernelName,
                                       CG->MOSModuleHandle, RawEvents, OutEvent,
                                       nullptr /* TODO: Pass mem allocation func
                                                  for accessors */
                                       ,
                                       PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT /* TODO: Extract from handler*/);
      if (Res != pi_result::PI_SUCCESS) {
        throw sycl::exception(
            sycl::errc::kernel,
            "Error during emulated graph command group submission.");
      }
      ScheduledEvents.push_back(NewEvent);
    } else {

      sycl::detail::EventImplPtr EventImpl =
          sycl::detail::Scheduler::getInstance().addCG(
              std::move(NodeImpl->getCGCopy()), Queue);

      ScheduledEvents.push_back(EventImpl);
    }
  }
  // Create an event which has all kernel events as dependencies
  sycl::detail::EventImplPtr NewEvent =
      std::make_shared<sycl::detail::event_impl>(Queue);
  NewEvent->setStateIncomplete();
  NewEvent->getPreparedDepsEvents() = ScheduledEvents;
#endif

  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  return QueueEvent;
}
} // namespace detail

template <>
command_graph<graph_state::modifiable>::command_graph(
    const sycl::context &SyclContext, const sycl::device &SyclDevice,
    const sycl::property_list &)
    : impl(std::make_shared<detail::graph_impl>(SyclContext, SyclDevice)) {}

template <>
node command_graph<graph_state::modifiable>::add_impl(
    const std::vector<node> &Deps) {
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  std::shared_ptr<detail::node_impl> NodeImpl = impl->add(DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

template <>
node command_graph<graph_state::modifiable>::add_impl(
    std::function<void(handler &)> CGF, const std::vector<node> &Deps) {
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  std::shared_ptr<detail::node_impl> NodeImpl =
      impl->add(impl, CGF, {}, DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

template <>
void command_graph<graph_state::modifiable>::make_edge(node &Src, node &Dest) {
  std::shared_ptr<detail::node_impl> SenderImpl =
      sycl::detail::getSyclObjImpl(Src);
  std::shared_ptr<detail::node_impl> ReceiverImpl =
      sycl::detail::getSyclObjImpl(Dest);

  SenderImpl->register_successor(ReceiverImpl,
                                 SenderImpl); // register successor
  impl->remove_root(ReceiverImpl); // remove receiver from root node list
}

template <>
command_graph<graph_state::executable>
command_graph<graph_state::modifiable>::finalize(
    const sycl::property_list &) const {
  return command_graph<graph_state::executable>{this->impl,
                                                this->impl->get_context()};
}

template <>
bool command_graph<graph_state::modifiable>::begin_recording(
    queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl->getCommandGraph() == nullptr) {
    QueueImpl->setCommandGraph(impl);
    impl->add_queue(QueueImpl);
    return true;
  } else if (QueueImpl->getCommandGraph() != impl) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "begin_recording called for a queue which is already "
                          "recording to a different graph.");
  }

  // Queue was already recording to this graph.
  return false;
}

template <>
bool command_graph<graph_state::modifiable>::begin_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (queue Queue : RecordingQueues) {
    QueueStateChanged |= this->begin_recording(Queue);
  }
  return QueueStateChanged;
}

template <> bool command_graph<graph_state::modifiable>::end_recording() {
  return impl->clear_queues();
}

template <>
bool command_graph<graph_state::modifiable>::end_recording(
    queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl->getCommandGraph() == impl) {
    QueueImpl->setCommandGraph(nullptr);
    impl->remove_queue(QueueImpl);
    return true;
  } else if (QueueImpl->getCommandGraph() != nullptr) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "end_recording called for a queue which is recording "
                          "to a different graph.");
  }

  // Queue was not recording to a graph.
  return false;
}

template <>
bool command_graph<graph_state::modifiable>::end_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (queue Queue : RecordingQueues) {
    QueueStateChanged |= this->end_recording(Queue);
  }
  return QueueStateChanged;
}

command_graph<graph_state::executable>::command_graph(
    const std::shared_ptr<detail::graph_impl> &Graph, const sycl::context &Ctx)
    : MTag(rand()),
      impl(std::make_shared<detail::exec_graph_impl>(Ctx, Graph)) {
  finalize_impl(); // Create backend representation for executable graph
}

void command_graph<graph_state::executable>::finalize_impl() {
  // Create PI command-buffers for each device in the finalized context
  impl->schedule();
#if SYCL_EXT_ONEAPI_GRAPH
  for (auto device : impl->get_context().get_devices()) {
    impl->create_pi_command_buffers(device);
  }
#endif
}

void command_graph<graph_state::executable>::update(
    const command_graph<graph_state::modifiable> &Graph) {
  (void)Graph;
  throw sycl::exception(make_error_code(errc::invalid),
                        "Method not yet implemented");
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
