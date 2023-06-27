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

/// Recursive check if a graph node or its successors contains a given kernel
/// argument.
/// @param[in] Arg The kernel argument to check for.
/// @param[in] CurrentNode The current graph node being checked.
/// @param[in,out] Deps The unique list of dependencies which have been
/// identified for this arg.
/// @return True if a dependency was added in this node or any of its
/// successors.
bool checkForArg(const sycl::detail::ArgDesc &Arg,
                 const std::shared_ptr<node_impl> &CurrentNode,
                 std::set<std::shared_ptr<node_impl>> &Deps) {
  bool SuccessorAddedDep = false;
  for (auto &Successor : CurrentNode->MSuccessors) {
    SuccessorAddedDep |= checkForArg(Arg, Successor, Deps);
  }

  if (!CurrentNode->isEmpty() && Deps.find(CurrentNode) == Deps.end() &&
      CurrentNode->hasArg(Arg) && !SuccessorAddedDep) {
    Deps.insert(CurrentNode);
    return true;
  }
  return SuccessorAddedDep;
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
  if (Handler.MCGType == sycl::detail::CG::None) {
    return this->add(Dep);
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
        checkForArg(Arg, NodePtr, UniqueDeps);
      }
    }

    // Add any deps determined from accessor arguments into the dependency list
    Deps.insert(Deps.end(), UniqueDeps.begin(), UniqueDeps.end());
  }

  // Add any nodes specified by event dependencies into the dependency list
  for (auto Dep : CommandGroup->getEvents()) {
    if (auto NodeImpl = MEventsMap.find(Dep); NodeImpl != MEventsMap.end()) {
      Deps.push_back(NodeImpl->second);
    } else {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Event dependency from handler::depends_on does "
                            "not correspond to a node within the graph");
    }
  }

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

exec_graph_impl::~exec_graph_impl() { MSchedule.clear(); }

sycl::event exec_graph_impl::enqueue(
    const std::shared_ptr<sycl::detail::queue_impl> &Queue) {
  std::vector<sycl::detail::pi::PiEvent> RawEvents;
  auto CreateNewEvent([&]() {
    auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
    NewEvent->setContextImpl(Queue->getContextImplPtr());
    NewEvent->setStateIncomplete();
    return NewEvent;
  });

  sycl::detail::EventImplPtr NewEvent;

  {
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

  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  return QueueEvent;
}

modifiable_command_graph::modifiable_command_graph(
    const sycl::context &SyclContext, const sycl::device &SyclDevice,
    const sycl::property_list &)
    : impl(std::make_shared<detail::graph_impl>(SyclContext, SyclDevice)) {}

node modifiable_command_graph::addImpl(const std::vector<node> &Deps) {
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  std::shared_ptr<detail::node_impl> NodeImpl = impl->add(DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

node modifiable_command_graph::addImpl(std::function<void(handler &)> CGF,
                                       const std::vector<node> &Deps) {
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

  SenderImpl->registerSuccessor(ReceiverImpl,
                                SenderImpl); // register successor
  impl->removeRoot(ReceiverImpl); // remove receiver from root node list
}

command_graph<graph_state::executable>
modifiable_command_graph::finalize(const sycl::property_list &) const {
  return command_graph<graph_state::executable>{this->impl,
                                                this->impl->getContext()};
}

bool modifiable_command_graph::begin_recording(queue &RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
