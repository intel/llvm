//==--------- graph_impl.cpp - SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/graph_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
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

void exec_graph_impl::schedule() {
  if (MSchedule.empty()) {
    for (auto Node : MGraphImpl->MRoots) {
      Node->topology_sort(Node, MSchedule);
    }
  }
}

// Recursively check if a given node is an exit node and add the new nodes as
// successors if so.
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

// Recursive check if a graph node or its successors contains a given kernel
// argument.
//
// @param[in] Arg The kernel argument to check for.
// @param[in] CurrentNode The current graph node being checked.
// @param[in,out] Deps The unique list of dependencies which have been
// identified for this arg.
//
// @returns True if a dependency was added in this node of any of its
// successors.
bool check_for_arg(const sycl::detail::ArgDesc &Arg,
                   const std::shared_ptr<node_impl> &CurrentNode,
                   std::set<std::shared_ptr<node_impl>> &Deps) {
  bool SuccessorAddedDep = false;
  for (auto &Successor : CurrentNode->MSuccessors) {
    SuccessorAddedDep |= check_for_arg(Arg, Successor, Deps);
  }

  if (Deps.find(CurrentNode) == Deps.end() && CurrentNode->has_arg(Arg) &&
      !SuccessorAddedDep) {
    Deps.insert(CurrentNode);
    return true;
  }
  return SuccessorAddedDep;
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

  // If the handler recorded a subgraph return that here as the relevant nodes
  // have already been added. The node returned here is an empty node with
  // dependencies on all the exit nodes of the subgraph.
  if (Handler.MSubgraphNode) {
    return Handler.MSubgraphNode;
  }
  // TODO: Do we need to pass event dependencies here for the explicit API?
  return this->add(Handler.MKernel, Handler.MNDRDesc, Handler.MOSModuleHandle,
                   Handler.MKernelName, Handler.MAccStorage,
                   Handler.MLocalAccStorage, Handler.MRequirements,
                   Handler.MArgs, Dep);
}

std::shared_ptr<node_impl> graph_impl::add(
    std::shared_ptr<sycl::detail::kernel_impl> Kernel,
    sycl::detail::NDRDescT NDRDesc, sycl::detail::OSModuleHandle OSModuleHandle,
    std::string KernelName,
    const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
    const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
    const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
    const std::vector<sycl::detail::ArgDesc> &Args,
    const std::vector<std::shared_ptr<node_impl>> &Dep,
    const std::vector<std::shared_ptr<sycl::detail::event_impl>> &DepEvents) {
  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>(
      Kernel, NDRDesc, OSModuleHandle, KernelName, AccStorage, LocalAccStorage,
      Requirements, Args);
  // Copy deps so we can modify them
  auto Deps = Dep;
  // A unique set of dependencies obtained by checking kernel arguments
  // for accessors
  std::set<std::shared_ptr<node_impl>> UniqueDeps;
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

  // Add any nodes specified by event dependencies into the dependency list
  for (auto Dep : DepEvents) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      Deps.push_back(NodeImpl->second);
    } else {
      throw sycl::exception(errc::invalid,
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    }
  }

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
void exec_graph_impl::find_real_deps(std::vector<pi_ext_sync_point> &Deps,
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

void exec_graph_impl::create_pi_command_buffers(sycl::device D) {
  // TODO we only have a single command-buffer per graph here, but
  // this will need to be multiple command-buffers for non-trivial graphs
  pi_ext_command_buffer OutCommandBuffer;
  pi_ext_command_buffer_desc Desc{};
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
    pi_kernel PiKernel = nullptr;
    std::mutex *KernelMutex = nullptr;
    pi_program PiProgram = nullptr;

    auto Kernel = Node->MKernel;
    const sycl::detail::KernelArgMask *EliminatedArgMask;
    if (Kernel != nullptr) {
      PiKernel = Kernel->getHandleRef();
    } else {
      std::tie(PiKernel, KernelMutex, EliminatedArgMask, PiProgram) =
          sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
              Node->MOSModuleHandle, ContextImpl, DeviceImpl, Node->MKernelName,
              nullptr);
    }

    auto SetFunc = [&Plugin, &PiKernel, this](sycl::detail::ArgDesc &Arg,
                                              size_t NextTrueIndex) {
      sycl::detail::SetArgBasedOnType(
          Plugin, PiKernel,
          nullptr /* TODO: Handle spec constants and pass device image here */,
          nullptr /* TODO: Pass getMemAllocation function for buffers */,
          this->MContext, false, Arg, NextTrueIndex);
    };
    std::vector<sycl::detail::ArgDesc> Args;
    sycl::detail::applyFuncOnFilteredArgs(EliminatedArgMask, Node->MArgs,
                                          SetFunc);

    std::vector<pi_ext_sync_point> Deps;
    for (auto &N : Node->MPredecessors) {
      find_real_deps(Deps, N.lock());
    }

    // add commands
    // Remember this information before the range dimensions are reversed
    const bool HasLocalSize = (Node->MNDRDesc.LocalSize[0] != 0);

    // Reverse kernel dims
    auto NDRDesc = Node->MNDRDesc;
    sycl::detail::ReverseRangeDimensionsForKernel(NDRDesc);

    size_t RequiredWGSize[3] = {0, 0, 0};
    size_t *LocalSize = nullptr;

    if (HasLocalSize)
      LocalSize = &NDRDesc.LocalSize[0];
    else {
      Plugin.call<sycl::detail::PiApiKind::piKernelGetGroupInfo>(
          PiKernel, DeviceImpl->getHandleRef(),
          PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
          RequiredWGSize, /* param_value_size_ret = */ nullptr);

      const bool EnforcedLocalSize =
          (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
           RequiredWGSize[2] != 0);
      if (EnforcedLocalSize)
        LocalSize = RequiredWGSize;
    }

    pi_ext_sync_point NewSyncPoint;
    Res = Plugin.call_nocheck<
        sycl::detail::PiApiKind::piextCommandBufferNDRangeKernel>(
        OutCommandBuffer, PiKernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
        &NDRDesc.GlobalSize[0], LocalSize, Deps.size(),
        Deps.size() ? Deps.data() : nullptr, &NewSyncPoint);

    if (Res != pi_result::PI_SUCCESS) {
      throw sycl::exception(errc::invalid,
                            "Failed to add kernel to PI command-buffer");
    }

    // Associate the new syncpoint with the current node
    MPiSyncPoints[Node] = NewSyncPoint;
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
  std::vector<pi_event> RawEvents;
  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });
#if SYCL_EXT_ONEAPI_GRAPH
  auto NewEvent = CreateNewEvent();
  pi_event *OutEvent = &NewEvent->getHandleRef();
  auto CommandBuffer = MPiCommandBuffers[Queue->get_device()];
  pi_result Res =
      Queue->getPlugin()
          .call_nocheck<sycl::detail::PiApiKind::piextEnqueueCommandBuffer>(
              CommandBuffer, Queue->getHandleRef(), RawEvents.size(),
              RawEvents.empty() ? nullptr : &RawEvents[0], OutEvent);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::event,
                          "Failed to enqueue event for node submission");
  }

#else
  std::vector<std::shared_ptr<sycl::detail::event_impl>> ScheduledEvents;
  for (auto &NodeImpl : MSchedule) {
    std::vector<RT::PiEvent> RawEvents;
    auto NewEvent = CreateNewEvent();
    pi_event *OutEvent = &NewEvent->getHandleRef();
    pi_int32 Res = sycl::detail::enqueueImpKernel(
        Queue, NodeImpl->MNDRDesc, NodeImpl->MArgs,
        nullptr /* TODO: Handle KernelBundles */, NodeImpl->MKernel,
        NodeImpl->MKernelName, NodeImpl->MOSModuleHandle, RawEvents, OutEvent,
        nullptr /* TODO: Pass mem allocation func for accessors */,
        PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT /* TODO: Extract from handler*/);
    if (Res != pi_result::PI_SUCCESS) {
      throw sycl::exception(
          sycl::errc::kernel,
          "Error during emulated graph command group submission.");
    }
    ScheduledEvents.push_back(NewEvent);
  }
  // Create an event which has all kernel events as dependencies
  auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
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
    const sycl::context &syclContext, const sycl::device &syclDevice,
    const sycl::property_list &)
    : impl(std::make_shared<detail::graph_impl>(syclContext, syclDevice)) {}

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
void command_graph<graph_state::modifiable>::make_edge(node Sender,
                                                       node Receiver) {
  std::shared_ptr<detail::node_impl> SenderImpl =
      sycl::detail::getSyclObjImpl(Sender);
  std::shared_ptr<detail::node_impl> ReceiverImpl =
      sycl::detail::getSyclObjImpl(Receiver);

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
    queue RecordingQueue) {
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
  for (auto &Queue : RecordingQueues) {
    QueueStateChanged |= this->begin_recording(Queue);
  }
  return QueueStateChanged;
}

template <> bool command_graph<graph_state::modifiable>::end_recording() {
  return impl->clear_queues();
}

template <>
bool command_graph<graph_state::modifiable>::end_recording(
    queue RecordingQueue) {
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
  for (auto &Queue : RecordingQueues) {
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

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
