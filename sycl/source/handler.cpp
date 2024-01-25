//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/graph_impl.hpp>
#include <detail/handler_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/usm/usm_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stream.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {

bool isDeviceGlobalUsedInKernel(const void *DeviceGlobalPtr) {
  DeviceGlobalMapEntry *DGEntry =
      detail::ProgramManager::getInstance().getDeviceGlobalEntry(
          DeviceGlobalPtr);
  return DGEntry && !DGEntry->MImageIdentifiers.empty();
}

sycl::detail::pi::PiImageCopyFlags
getPiImageCopyFlags(sycl::usm::alloc SrcPtrType, sycl::usm::alloc DstPtrType) {
  if (DstPtrType == sycl::usm::alloc::device) {
    // Dest is on device
    if (SrcPtrType == sycl::usm::alloc::device)
      return sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_DEVICE;
    if (SrcPtrType == sycl::usm::alloc::host ||
        SrcPtrType == sycl::usm::alloc::unknown)
      return sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE;
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unknown copy source location");
  }
  if (DstPtrType == sycl::usm::alloc::host ||
      DstPtrType == sycl::usm::alloc::unknown) {
    // Dest is on host
    if (SrcPtrType == sycl::usm::alloc::device)
      return sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST;
    if (SrcPtrType == sycl::usm::alloc::host ||
        SrcPtrType == sycl::usm::alloc::unknown)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Cannot copy image from host to host");
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unknown copy source location");
  }
  throw sycl::exception(make_error_code(errc::invalid),
                        "Unknown copy destination location");
}

} // namespace detail

handler::handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
    : handler(Queue, Queue, nullptr, IsHost) {}

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 std::shared_ptr<detail::queue_impl> PrimaryQueue,
                 std::shared_ptr<detail::queue_impl> SecondaryQueue,
                 bool IsHost)
    : MImpl(std::make_shared<detail::handler_impl>(std::move(PrimaryQueue),
                                                   std::move(SecondaryQueue))),
      MQueue(std::move(Queue)), MIsHost(IsHost) {}

handler::handler(
    std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph)
    : MImpl(std::make_shared<detail::handler_impl>()), MGraph(Graph) {}

// Sets the submission state to indicate that an explicit kernel bundle has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that a specialization constant has been set.
void handler::setStateExplicitKernelBundle() {
  MImpl->setStateExplicitKernelBundle();
}

// Sets the submission state to indicate that a specialization constant has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that an explicit kernel bundle has been set.
void handler::setStateSpecConstSet() { MImpl->setStateSpecConstSet(); }

// Returns true if the submission state is EXPLICIT_KERNEL_BUNDLE_STATE and
// false otherwise.
bool handler::isStateExplicitKernelBundle() const {
  return MImpl->isStateExplicitKernelBundle();
}

// Returns a shared_ptr to the kernel_bundle.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns shared_ptr(nullptr) if Insert is false
std::shared_ptr<detail::kernel_bundle_impl>
handler::getOrInsertHandlerKernelBundle(bool Insert) const {
  if (!MImpl->MKernelBundle && Insert) {
    auto Ctx = MGraph ? MGraph->getContext() : MQueue->get_context();
    auto Dev = MGraph ? MGraph->getDevice() : MQueue->get_device();
    MImpl->MKernelBundle = detail::getSyclObjImpl(
        get_kernel_bundle<bundle_state::input>(Ctx, {Dev}, {}));
  }
  return MImpl->MKernelBundle;
}

// Sets kernel bundle to the provided one.
void handler::setHandlerKernelBundle(
    const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr) {
  MImpl->MKernelBundle = NewKernelBundleImpPtr;
}

void handler::setHandlerKernelBundle(kernel Kernel) {
  // Kernel may not have an associated kernel bundle if it is created from a
  // program. As such, apply getSyclObjImpl directly on the kernel, i.e. not
  //  the other way around: getSyclObjImp(Kernel->get_kernel_bundle()).
  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpl =
      detail::getSyclObjImpl(Kernel)->get_kernel_bundle();
  setHandlerKernelBundle(KernelBundleImpl);
}

event handler::finalize() {
  // This block of code is needed only for reduction implementation.
  // It is harmless (does nothing) for everything else.
  if (MIsFinalized)
    return MLastEvent;
  MIsFinalized = true;

  // If we have a subgraph node that means that a subgraph was recorded as
  // part of this queue submission, so we skip adding a new node here since
  // they have already been added, and return the event associated with the
  // subgraph node.
  if (MQueue && MQueue->getCommandGraph() && MSubgraphNode) {
    return detail::createSyclObjFromImpl<event>(
        MQueue->getCommandGraph()->getEventForNode(MSubgraphNode));
  }

  // According to 4.7.6.9 of SYCL2020 spec, if a placeholder accessor is passed
  // to a command without being bound to a command group, an exception should
  // be thrown.
  {
    for (const auto &arg : MArgs) {
      if (arg.MType != detail::kernel_param_kind_t::kind_accessor)
        continue;

      detail::Requirement *AccImpl =
          static_cast<detail::Requirement *>(arg.MPtr);
      if (AccImpl->MIsPlaceH) {
        auto It = std::find(CGData.MRequirements.begin(),
                            CGData.MRequirements.end(), AccImpl);
        if (It == CGData.MRequirements.end())
          throw sycl::exception(make_error_code(errc::kernel_argument),
                                "placeholder accessor must be bound by calling "
                                "handler::require() before it can be used.");
      }
    }
  }

  const auto &type = getType();
  if (type == detail::CG::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/false);
    if (KernelBundleImpPtr) {
      // Make sure implicit non-interop kernel bundles have the kernel
      if (!KernelBundleImpPtr->isInterop() &&
          !MImpl->isStateExplicitKernelBundle()) {
        auto Dev = MGraph ? MGraph->getDevice() : MQueue->get_device();
        kernel_id KernelID =
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
            detail::ProgramManager::getInstance().getSYCLKernelID(
                MKernelName.c_str());
#else
            detail::ProgramManager::getInstance().getSYCLKernelID(MKernelName);
#endif
        bool KernelInserted = KernelBundleImpPtr->add_kernel(KernelID, Dev);
        // If kernel was not inserted and the bundle is in input mode we try
        // building it and trying to find the kernel in executable mode
        if (!KernelInserted &&
            KernelBundleImpPtr->get_bundle_state() == bundle_state::input) {
          auto KernelBundle =
              detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
                  KernelBundleImpPtr);
          kernel_bundle<bundle_state::executable> ExecKernelBundle =
              build(KernelBundle);
          KernelBundleImpPtr = detail::getSyclObjImpl(ExecKernelBundle);
          setHandlerKernelBundle(KernelBundleImpPtr);
          KernelInserted = KernelBundleImpPtr->add_kernel(KernelID, Dev);
        }
        // If the kernel was not found in executable mode we throw an exception
        if (!KernelInserted)
          throw sycl::exception(make_error_code(errc::runtime),
                                "Failed to add kernel to kernel bundle.");
      }

      switch (KernelBundleImpPtr->get_bundle_state()) {
      case bundle_state::input: {
        // Underlying level expects kernel_bundle to be in executable state
        kernel_bundle<bundle_state::executable> ExecBundle = build(
            detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
                KernelBundleImpPtr));
        KernelBundleImpPtr = detail::getSyclObjImpl(ExecBundle);
        setHandlerKernelBundle(KernelBundleImpPtr);
        break;
      }
      case bundle_state::executable:
        // Nothing to do
        break;
      case bundle_state::object:
      case bundle_state::ext_oneapi_source:
        assert(0 && "Expected that the bundle is either in input or executable "
                    "states.");
        break;
      }
    }

    if (MQueue && !MGraph && !MSubgraphNode && !MQueue->getCommandGraph() &&
        !MQueue->is_in_fusion_mode() &&
        CGData.MRequirements.size() + CGData.MEvents.size() +
                MStreamStorage.size() ==
            0) {
      // if user does not add a new dependency to the dependency graph, i.e.
      // the graph is not changed, and the queue is not in fusion mode, then
      // this faster path is used to submit kernel bypassing scheduler and
      // avoiding CommandGroup, Command objects creation.

      std::vector<sycl::detail::pi::PiEvent> RawEvents;
      detail::EventImplPtr NewEvent;

#ifdef XPTI_ENABLE_INSTRUMENTATION
      // uint32_t StreamID, uint64_t InstanceID, xpti_td* TraceEvent,
      int32_t StreamID = xptiRegisterStream(detail::SYCL_STREAM_NAME);
      auto [CmdTraceEvent, InstanceID] = emitKernelInstrumentationData(
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
          StreamID, MKernel, MCodeLoc, MKernelName.c_str(), MQueue, MNDRDesc,
#else
          StreamID, MKernel, MCodeLoc, MKernelName, MQueue, MNDRDesc,
#endif
          KernelBundleImpPtr, MArgs);
      auto EnqueueKernel = [&, CmdTraceEvent = CmdTraceEvent,
                            InstanceID = InstanceID]() {
#else
      auto EnqueueKernel = [&]() {
#endif
        // 'Result' for single point of return
        pi_int32 Result = PI_ERROR_INVALID_VALUE;
#ifdef XPTI_ENABLE_INSTRUMENTATION
        detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                           xpti::trace_task_begin, nullptr);
#endif
        if (MQueue->is_host()) {
          MHostKernel->call(MNDRDesc, (NewEvent)
                                          ? NewEvent->getHostProfilingInfo()
                                          : nullptr);
          Result = PI_SUCCESS;
        } else {
          if (MQueue->getDeviceImplPtr()->getBackend() ==
              backend::ext_intel_esimd_emulator) {
            // Capture the host timestamp for profiling (queue time)
            if (NewEvent != nullptr)
              NewEvent->setHostEnqueueTime();
            MQueue->getPlugin()->call<detail::PiApiKind::piEnqueueKernelLaunch>(
                nullptr, reinterpret_cast<pi_kernel>(MHostKernel->getPtr()),
                MNDRDesc.Dims, &MNDRDesc.GlobalOffset[0],
                &MNDRDesc.GlobalSize[0], &MNDRDesc.LocalSize[0], 0, nullptr,
                nullptr);
            Result = PI_SUCCESS;
          } else {
            Result =
                enqueueImpKernel(MQueue, MNDRDesc, MArgs, KernelBundleImpPtr,
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
                                 MKernel, MKernelName.c_str(), RawEvents,
                                 NewEvent, nullptr, MImpl->MKernelCacheConfig);
#else
                                 MKernel, MKernelName, RawEvents, NewEvent,
                                 nullptr, MImpl->MKernelCacheConfig);
#endif
          }
        }
#ifdef XPTI_ENABLE_INSTRUMENTATION
        // Emit signal only when event is created
        if (NewEvent != nullptr) {
          detail::emitInstrumentationGeneral(
              StreamID, InstanceID, CmdTraceEvent, xpti::trace_signal,
              static_cast<const void *>(NewEvent->getHandleRef()));
        }
        detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                           xpti::trace_task_end, nullptr);
#endif
        return Result;
      };

      bool DiscardEvent = false;
      if (MQueue->has_discard_events_support()) {
        // Kernel only uses assert if it's non interop one
        bool KernelUsesAssert =
            !(MKernel && MKernel->isInterop()) &&
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
            detail::ProgramManager::getInstance().kernelUsesAssert(
                MKernelName.c_str());
#else
            detail::ProgramManager::getInstance().kernelUsesAssert(MKernelName);
#endif
        DiscardEvent = !KernelUsesAssert;
      }

      if (DiscardEvent) {
        if (PI_SUCCESS != EnqueueKernel())
          throw runtime_error("Enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
      } else {
        NewEvent = std::make_shared<detail::event_impl>(MQueue);
        NewEvent->setContextImpl(MQueue->getContextImplPtr());
        NewEvent->setStateIncomplete();
        NewEvent->setSubmissionTime();

        if (PI_SUCCESS != EnqueueKernel())
          throw runtime_error("Enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
        else if (NewEvent->is_host() || NewEvent->getHandleRef() == nullptr)
          NewEvent->setComplete();

        MLastEvent = detail::createSyclObjFromImpl<event>(NewEvent);
      }
      return MLastEvent;
    }
  }

  std::unique_ptr<detail::CG> CommandGroup;
  switch (type) {
  case detail::CG::Kernel: {
    // Copy kernel name here instead of move so that it's available after
    // running of this method by reductions implementation. This allows for
    // assert feature to check if kernel uses assertions
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(MImpl->MKernelBundle), std::move(CGData), std::move(MArgs),
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
        MKernelName.c_str(), std::move(MStreamStorage),
#else
        MKernelName, std::move(MStreamStorage),
#endif
        std::move(MImpl->MAuxiliaryResources), MCGType,
        MImpl->MKernelCacheConfig, MCodeLoc));
    break;
  }
  case detail::CG::CopyAccToPtr:
  case detail::CG::CopyPtrToAcc:
  case detail::CG::CopyAccToAcc:
    CommandGroup.reset(
        new detail::CGCopy(MCGType, MSrcPtr, MDstPtr, std::move(CGData),
                           std::move(MImpl->MAuxiliaryResources), MCodeLoc));
    break;
  case detail::CG::Fill:
    CommandGroup.reset(new detail::CGFill(std::move(MPattern), MDstPtr,
                                          std::move(CGData), MCodeLoc));
    break;
  case detail::CG::UpdateHost:
    CommandGroup.reset(
        new detail::CGUpdateHost(MDstPtr, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::CopyUSM:
    CommandGroup.reset(new detail::CGCopyUSM(MSrcPtr, MDstPtr, MLength,
                                             std::move(CGData), MCodeLoc));
    break;
  case detail::CG::FillUSM:
    CommandGroup.reset(new detail::CGFillUSM(
        std::move(MPattern), MDstPtr, MLength, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::PrefetchUSM:
    CommandGroup.reset(new detail::CGPrefetchUSM(MDstPtr, MLength,
                                                 std::move(CGData), MCodeLoc));
    break;
  case detail::CG::AdviseUSM:
    CommandGroup.reset(new detail::CGAdviseUSM(MDstPtr, MLength, MImpl->MAdvice,
                                               std::move(CGData), MCGType,
                                               MCodeLoc));
    break;
  case detail::CG::Copy2DUSM:
    CommandGroup.reset(new detail::CGCopy2DUSM(
        MSrcPtr, MDstPtr, MImpl->MSrcPitch, MImpl->MDstPitch, MImpl->MWidth,
        MImpl->MHeight, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::Fill2DUSM:
    CommandGroup.reset(new detail::CGFill2DUSM(
        std::move(MPattern), MDstPtr, MImpl->MDstPitch, MImpl->MWidth,
        MImpl->MHeight, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::Memset2DUSM:
    CommandGroup.reset(new detail::CGMemset2DUSM(
        MPattern[0], MDstPtr, MImpl->MDstPitch, MImpl->MWidth, MImpl->MHeight,
        std::move(CGData), MCodeLoc));
    break;
  case detail::CG::CodeplayHostTask: {
    auto context = MGraph ? detail::getSyclObjImpl(MGraph->getContext())
                          : MQueue->getContextImplPtr();
    CommandGroup.reset(new detail::CGHostTask(
        std::move(MHostTask), MQueue, context, std::move(MArgs),
        std::move(CGData), MCGType, MCodeLoc));
    break;
  }
  case detail::CG::Barrier:
  case detail::CG::BarrierWaitlist: {
    if (auto GraphImpl = getCommandGraph(); GraphImpl != nullptr) {
      // if no event to wait for was specified, we add all exit
      // nodes/events of the graph
      if (MEventsWaitWithBarrier.size() == 0) {
        MEventsWaitWithBarrier = GraphImpl->getExitNodesEvents();
        // Graph-wide barriers take precedence over previous one.
        // We therefore remove the previous ones from ExtraDependencies list.
        // The current barrier is then added to this list in the graph_impl.
        std::vector<detail::EventImplPtr> EventsBarriers =
            GraphImpl->removeBarriersFromExtraDependencies();
        MEventsWaitWithBarrier.insert(std::end(MEventsWaitWithBarrier),
                                      std::begin(EventsBarriers),
                                      std::end(EventsBarriers));
      }
      CGData.MEvents.insert(std::end(CGData.MEvents),
                            std::begin(MEventsWaitWithBarrier),
                            std::end(MEventsWaitWithBarrier));
      // Barrier node is implemented as an empty node in Graph
      // but keep the barrier type to help managing dependencies
      MCGType = detail::CG::Barrier;
      CommandGroup.reset(
          new detail::CG(detail::CG::Barrier, std::move(CGData), MCodeLoc));
    } else {
      CommandGroup.reset(
          new detail::CGBarrier(std::move(MEventsWaitWithBarrier),
                                std::move(CGData), MCGType, MCodeLoc));
    }
    break;
  }
  case detail::CG::CopyToDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyToDeviceGlobal(
        MSrcPtr, MDstPtr, MImpl->MIsDeviceImageScoped, MLength, MImpl->MOffset,
        std::move(CGData), MCodeLoc));
    break;
  }
  case detail::CG::CopyFromDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyFromDeviceGlobal(
        MSrcPtr, MDstPtr, MImpl->MIsDeviceImageScoped, MLength, MImpl->MOffset,
        std::move(CGData), MCodeLoc));
    break;
  }
  case detail::CG::ReadWriteHostPipe: {
    CommandGroup.reset(new detail::CGReadWriteHostPipe(
        MImpl->HostPipeName, MImpl->HostPipeBlocking, MImpl->HostPipePtr,
        MImpl->HostPipeTypeSize, MImpl->HostPipeRead, std::move(CGData),
        MCodeLoc));
    break;
  }
  case detail::CG::ExecCommandBuffer:
    // If we have a subgraph node we don't want to actually execute this command
    // graph submission.
    if (!MSubgraphNode) {
      event GraphCompletionEvent =
          MExecGraph->enqueue(MQueue, std::move(CGData));
      MLastEvent = GraphCompletionEvent;
      return MLastEvent;
    }
    break;
  case detail::CG::CopyImage:
    CommandGroup.reset(new detail::CGCopyImage(
        MSrcPtr, MDstPtr, MImpl->MImageDesc, MImpl->MImageFormat,
        MImpl->MImageCopyFlags, MImpl->MSrcOffset, MImpl->MDestOffset,
        MImpl->MHostExtent, MImpl->MCopyExtent, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::SemaphoreWait:
    CommandGroup.reset(new detail::CGSemaphoreWait(
        MImpl->MInteropSemaphoreHandle, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::SemaphoreSignal:
    CommandGroup.reset(new detail::CGSemaphoreSignal(
        MImpl->MInteropSemaphoreHandle, std::move(CGData), MCodeLoc));
    break;
  case detail::CG::None:
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      std::cout << "WARNING: An empty command group is submitted." << std::endl;
    }

    // Empty nodes are handled by Graph like standard nodes
    // For Standard mode (non-graph),
    // empty nodes are not sent to the scheduler to save time
    if (MGraph || (MQueue && MQueue->getCommandGraph())) {
      CommandGroup.reset(
          new detail::CG(detail::CG::None, std::move(CGData), MCodeLoc));
    } else {
      detail::EventImplPtr Event = std::make_shared<sycl::detail::event_impl>();
      MLastEvent = detail::createSyclObjFromImpl<event>(Event);
      return MLastEvent;
    }
    break;
  }

  if (!MSubgraphNode && !CommandGroup)
    throw sycl::runtime_error(
        "Internal Error. Command group cannot be constructed.",
        PI_ERROR_INVALID_OPERATION);

  // If there is a graph associated with the handler we are in the explicit
  // graph mode, so we store the CG instead of submitting it to the scheduler,
  // so it can be retrieved by the graph later.
  if (MGraph) {
    MGraphNodeCG = std::move(CommandGroup);
    return detail::createSyclObjFromImpl<event>(
        std::make_shared<detail::event_impl>());
  }

  // If the queue has an associated graph then we need to take the CG and pass
  // it to the graph to create a node, rather than submit it to the scheduler.
  if (auto GraphImpl = MQueue->getCommandGraph(); GraphImpl) {
    auto EventImpl = std::make_shared<detail::event_impl>();
    std::shared_ptr<ext::oneapi::experimental::detail::node_impl> NodeImpl =
        nullptr;

    // GraphImpl is read and written in this scope so we lock this graph
    // with full priviledges.
    ext::oneapi::experimental::detail::graph_impl::WriteLock Lock(
        GraphImpl->MMutex);

    // Create a new node in the graph representing this command-group
    if (MQueue->isInOrder()) {
      // In-order queues create implicit linear dependencies between nodes.
      // Find the last node added to the graph from this queue, so our new
      // node can set it as a predecessor.
      auto DependentNode = GraphImpl->getLastInorderNode(MQueue);

      NodeImpl = DependentNode
                     ? GraphImpl->add(MCGType, std::move(CommandGroup),
                                      {DependentNode})
                     : GraphImpl->add(MCGType, std::move(CommandGroup));

      // If we are recording an in-order queue remember the new node, so it
      // can be used as a dependency for any more nodes recorded from this
      // queue.
      GraphImpl->setLastInorderNode(MQueue, NodeImpl);
    } else {
      NodeImpl = GraphImpl->add(MCGType, std::move(CommandGroup));
    }

    // Associate an event with this new node and return the event.
    GraphImpl->addEventForNode(EventImpl, NodeImpl);

    EventImpl->setCommandGraph(GraphImpl);

    return detail::createSyclObjFromImpl<event>(EventImpl);
  }

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);
  return MLastEvent;
}

void handler::addReduction(const std::shared_ptr<const void> &ReduObj) {
  MImpl->MAuxiliaryResources.push_back(ReduObj);
}

void handler::associateWithHandlerCommon(detail::AccessorImplPtr AccImpl,
                                         int AccTarget) {
  if (getCommandGraph() &&
      static_cast<detail::SYCLMemObjT *>(AccImpl->MSYCLMemObj)
          ->needsWriteBack()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Accessors to buffers which have write_back enabled "
                          "are not allowed to be used in command graphs.");
  }
  detail::Requirement *Req = AccImpl.get();
  if (Req->MAccessMode != sycl::access_mode::read) {
    auto SYCLMemObj = static_cast<detail::SYCLMemObjT *>(Req->MSYCLMemObj);
    SYCLMemObj->handleWriteAccessorCreation();
  }
  // Add accessor to the list of requirements.
  if (Req->MAccessRange.size() != 0)
    CGData.MRequirements.push_back(Req);
  // Store copy of the accessor.
  CGData.MAccStorage.push_back(std::move(AccImpl));
  // Add an accessor to the handler list of associated accessors.
  // For associated accessors index does not means nothing.
  MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                   Req, AccTarget, /*index*/ 0);
}

void handler::associateWithHandler(detail::AccessorBaseHost *AccBase,
                                   access::target AccTarget) {
  associateWithHandlerCommon(detail::getSyclObjImpl(*AccBase),
                             static_cast<int>(AccTarget));
}

void handler::associateWithHandler(
    detail::UnsampledImageAccessorBaseHost *AccBase, image_target AccTarget) {
  associateWithHandlerCommon(detail::getSyclObjImpl(*AccBase),
                             static_cast<int>(AccTarget));
}

void handler::associateWithHandler(
    detail::SampledImageAccessorBaseHost *AccBase, image_target AccTarget) {
  associateWithHandlerCommon(detail::getSyclObjImpl(*AccBase),
                             static_cast<int>(AccTarget));
}

static void addArgsForGlobalAccessor(detail::Requirement *AccImpl, size_t Index,
                                     size_t &IndexShift, int Size,
                                     bool IsKernelCreatedFromSource,
                                     size_t GlobalSize,
                                     std::vector<detail::ArgDesc> &Args,
                                     bool isESIMD) {
  using detail::kernel_param_kind_t;
  if (AccImpl->PerWI)
    AccImpl->resize(GlobalSize);

  Args.emplace_back(kernel_param_kind_t::kind_accessor, AccImpl, Size,
                    Index + IndexShift);

  // TODO ESIMD currently does not suport offset, memory and access ranges -
  // accessor::init for ESIMD-mode accessor has a single field, translated
  // to a single kernel argument set above.
  if (!isESIMD && !IsKernelCreatedFromSource) {
    // Dimensionality of the buffer is 1 when dimensionality of the
    // accessor is 0.
    const size_t SizeAccField =
        sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MAccessRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MMemoryRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MOffset[0], SizeAccField, Index + IndexShift);
  }
}

void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource, bool IsESIMD) {
  using detail::kernel_param_kind_t;

  switch (Kind) {
  case kernel_param_kind_t::kind_std_layout:
  case kernel_param_kind_t::kind_pointer: {
    MArgs.emplace_back(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_stream: {
    // Stream contains several accessors inside.
    stream *S = static_cast<stream *>(Ptr);

    detail::AccessorBaseHost *GBufBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalBuf);
    detail::AccessorImplPtr GBufImpl = detail::getSyclObjImpl(*GBufBase);
    detail::Requirement *GBufReq = GBufImpl.get();
    addArgsForGlobalAccessor(GBufReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource,
                             MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GOffsetBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalOffset);
    detail::AccessorImplPtr GOfssetImpl = detail::getSyclObjImpl(*GOffsetBase);
    detail::Requirement *GOffsetReq = GOfssetImpl.get();
    addArgsForGlobalAccessor(GOffsetReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource,
                             MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GFlushBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalFlushBuf);
    detail::AccessorImplPtr GFlushImpl = detail::getSyclObjImpl(*GFlushBase);
    detail::Requirement *GFlushReq = GFlushImpl.get();

    size_t GlobalSize = MNDRDesc.GlobalSize.size();
    // If work group size wasn't set explicitly then it must be recieved
    // from kernel attribute or set to default values.
    // For now we can't get this attribute here.
    // So we just suppose that WG size is always default for stream.
    // TODO adjust MNDRDesc when device image contains kernel's attribute
    if (GlobalSize == 0) {
      // Suppose that work group size is 1 for every dimension
      GlobalSize = MNDRDesc.NumWorkGroups.size();
    }
    addArgsForGlobalAccessor(GFlushReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, MArgs,
                             IsESIMD);
    ++IndexShift;
    MArgs.emplace_back(kernel_param_kind_t::kind_std_layout,
                       &S->FlushBufferSize, sizeof(S->FlushBufferSize),
                       Index + IndexShift);

    break;
  }
  case kernel_param_kind_t::kind_accessor: {
    // For args kind of accessor Size is information about accessor.
    // The first 11 bits of Size encodes the accessor target.
    const access::target AccTarget =
        static_cast<access::target>(Size & AccessTargetMask);
    switch (AccTarget) {
    case access::target::device:
    case access::target::constant_buffer: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      addArgsForGlobalAccessor(AccImpl, Index, IndexShift, Size,
                               IsKernelCreatedFromSource,
                               MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAcc =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      range<3> &Size = LAcc->MSize;
      const int Dims = LAcc->MDims;
      int SizeInBytes = LAcc->MElemSize;
      for (int I = 0; I < Dims; ++I)
        SizeInBytes *= Size[I];
      // Some backends do not accept zero-sized local memory arguments, so we
      // make it a minimum allocation of 1 byte.
      SizeInBytes = std::max(SizeInBytes, 1);
      MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                         SizeInBytes, Index + IndexShift);
      // TODO ESIMD currently does not suport MSize field passing yet
      // accessor::init for ESIMD-mode accessor has a single field, translated
      // to a single kernel argument set above.
      if (!IsESIMD && !IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = Dims * sizeof(Size[0]);
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
      }
      break;
    }
    case access::target::image:
    case access::target::image_array: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // TODO Handle additional kernel arguments for image class
        // if the compiler front-end adds them.
      }
      break;
    }
    case access::target::host_image:
    case access::target::host_task:
    case access::target::host_buffer: {
      throw sycl::invalid_parameter_error("Unsupported accessor target case.",
                                          PI_ERROR_INVALID_OPERATION);
      break;
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    MArgs.emplace_back(kernel_param_kind_t::kind_sampler, Ptr, sizeof(sampler),
                       Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    MArgs.emplace_back(
        kernel_param_kind_t::kind_specialization_constants_buffer, Ptr, Size,
        Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw runtime_error("Invalid kernel param kind", PI_ERROR_INVALID_VALUE);
    break;
  }
}

// The argument can take up more space to store additional information about
// MAccessRange, MMemoryRange, and MOffset added with addArgsForGlobalAccessor.
// We use the worst-case estimate because the lifetime of the vector is short.
// In processArg the kind_stream case introduces the maximum number of
// additional arguments. The case adds additional 12 arguments to the currently
// processed argument, hence worst-case estimate is 12+1=13.
// TODO: the constant can be removed if the size of MArgs will be calculated at
// compile time.
inline constexpr size_t MaxNumAdditionalArgs = 13;

void handler::extractArgsAndReqs() {
  assert(MKernel && "MKernel is not initialized");
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
  MArgs.clear();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  const bool IsKernelCreatedFromSource = MKernel->isCreatedFromSource();
  MArgs.reserve(MaxNumAdditionalArgs * UnPreparedArgs.size());

  size_t IndexShift = 0;
  for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
    void *Ptr = UnPreparedArgs[I].MPtr;
    const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
    const int &Size = UnPreparedArgs[I].MSize;
    const int Index = UnPreparedArgs[I].MIndex;
    processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource,
               false);
  }
}

void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs, bool IsESIMD) {
  const bool IsKernelCreatedFromSource = false;
  size_t IndexShift = 0;
  MArgs.reserve(MaxNumAdditionalArgs * KernelArgsNum);

  for (size_t I = 0; I < KernelArgsNum; ++I) {
    void *Ptr = LambdaPtr + KernelArgs[I].offset;
    const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
    const int &Size = KernelArgs[I].info;
    if (Kind == detail::kernel_param_kind_t::kind_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & AccessTargetMask);
      if ((AccTarget == access::target::device ||
           AccTarget == access::target::constant_buffer) ||
          (AccTarget == access::target::image ||
           AccTarget == access::target::image_array)) {
        detail::AccessorBaseHost *AccBase =
            static_cast<detail::AccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*AccBase).get();
      } else if (AccTarget == access::target::local) {
        detail::LocalAccessorBaseHost *LocalAccBase =
            static_cast<detail::LocalAccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*LocalAccBase).get();
      }
    }
    processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource,
               IsESIMD);
  }
}

// Calling methods of kernel_impl requires knowledge of class layout.
// As this is impossible in header, there's a function that calls necessary
// method inside the library and returns the result.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
detail::string handler::getKernelName() {
  return detail::string(MKernel->get_info<info::kernel::function_name>());
}
#else
std::string handler::getKernelName() {
  return MKernel->get_info<info::kernel::function_name>();
}
#endif

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
void handler::verifyUsedKernelBundleInternal(detail::string_view KernelName) {
#else
void handler::verifyUsedKernelBundle(const std::string &KernelName) {
#endif
  auto UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundle(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  if (!MImpl->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  device Dev =
      MGraph ? MGraph->getDevice() : detail::getDeviceFromHandler(*this);
  if (!UsedKernelBundleImplPtr->has_kernel(KernelID, Dev))
    throw sycl::exception(
        make_error_code(errc::kernel_not_supported),
        "The kernel bundle in use does not contain the kernel");
}

void handler::ext_oneapi_barrier(const std::vector<event> &WaitList) {
  throwIfActionIsCreated();
  MCGType = detail::CG::BarrierWaitlist;
  MEventsWaitWithBarrier.resize(WaitList.size());
  std::transform(
      WaitList.begin(), WaitList.end(), MEventsWaitWithBarrier.begin(),
      [](const event &Event) { return detail::getSyclObjImpl(Event); });
}

using namespace sycl::detail;
bool handler::DisableRangeRounding() {
  return SYCLConfig<SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING>::get();
}

bool handler::RangeRoundingTrace() {
  return SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE>::get();
}

void handler::GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                                       size_t &MinRange) {
  SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>::GetSettings(
      MinFactor, GoodFactor, MinRange);
}

void handler::memcpy(void *Dest, const void *Src, size_t Count) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  MLength = Count;
  setType(detail::CG::CopyUSM);
}

void handler::memset(void *Dest, int Value, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MLength = Count;
  setType(detail::CG::FillUSM);
}

void handler::prefetch(const void *Ptr, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  setType(detail::CG::PrefetchUSM);
}

void handler::mem_advise(const void *Ptr, size_t Count, int Advice) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  MImpl->MAdvice = static_cast<pi_mem_advice>(Advice);
  setType(detail::CG::AdviseUSM);
}

void handler::ext_oneapi_memcpy2d_impl(void *Dest, size_t DestPitch,
                                       const void *Src, size_t SrcPitch,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  MImpl->MSrcPitch = SrcPitch;
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Copy2DUSM);
}

void handler::ext_oneapi_fill2d_impl(void *Dest, size_t DestPitch,
                                     const void *Value, size_t ValueSize,
                                     size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.resize(ValueSize);
  std::memcpy(MPattern.data(), Value, ValueSize);
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Fill2DUSM);
}

void handler::ext_oneapi_memset2d_impl(void *Dest, size_t DestPitch, int Value,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Memset2DUSM);
}

void handler::ext_oneapi_copy(
    void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &Desc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src;
  MDstPtr = Dest.raw_handle;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(Desc.channel_order);

  MImpl->MSrcOffset = {0, 0, 0};
  MImpl->MDestOffset = {0, 0, 0};
  MImpl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MHostExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE;
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_copy(
    void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src;
  MDstPtr = Dest.raw_handle;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = DestImgDesc.width;
  PiDesc.image_height = DestImgDesc.height;
  PiDesc.image_depth = DestImgDesc.depth;
  PiDesc.image_type = DestImgDesc.depth > 0
                          ? PI_MEM_TYPE_IMAGE3D
                          : (DestImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                    : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(DestImgDesc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(DestImgDesc.channel_order);

  MImpl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  MImpl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  MImpl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  MImpl->MHostExtent = {SrcExtent[0], SrcExtent[1], SrcExtent[2]};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE;
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_copy(
    ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &Desc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src.raw_handle;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(Desc.channel_order);

  MImpl->MSrcOffset = {0, 0, 0};
  MImpl->MDestOffset = {0, 0, 0};
  MImpl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MHostExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST;
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_copy(
    ext::oneapi::experimental::image_mem_handle Src, sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src.raw_handle;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = SrcImgDesc.width;
  PiDesc.image_height = SrcImgDesc.height;
  PiDesc.image_depth = SrcImgDesc.depth;
  PiDesc.image_type =
      SrcImgDesc.depth > 0
          ? PI_MEM_TYPE_IMAGE3D
          : (SrcImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(SrcImgDesc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(SrcImgDesc.channel_order);

  MImpl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  MImpl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  MImpl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  MImpl->MHostExtent = {DestExtent[0], DestExtent[1], DestExtent[2]};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST;
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_copy(
    void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &Desc, size_t Pitch) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(Desc.channel_order);

  MImpl->MSrcOffset = {0, 0, 0};
  MImpl->MDestOffset = {0, 0, 0};
  MImpl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MHostExtent = {Desc.width, Desc.height, Desc.depth};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageDesc.image_row_pitch = Pitch;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags = detail::getPiImageCopyFlags(
      get_pointer_type(Src, MQueue->get_context()),
      get_pointer_type(Dest, MQueue->get_context()));
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_copy(
    void *Src, sycl::range<3> SrcOffset, void *Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = Src;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = DeviceImgDesc.width;
  PiDesc.image_height = DeviceImgDesc.height;
  PiDesc.image_depth = DeviceImgDesc.depth;
  PiDesc.image_type = DeviceImgDesc.depth > 0
                          ? PI_MEM_TYPE_IMAGE3D
                          : (DeviceImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                      : PI_MEM_TYPE_IMAGE1D);

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(DeviceImgDesc.channel_type);
  PiFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(DeviceImgDesc.channel_order);

  MImpl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  MImpl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  MImpl->MHostExtent = {HostExtent[0], HostExtent[1], HostExtent[2]};
  MImpl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  MImpl->MImageDesc = PiDesc;
  MImpl->MImageDesc.image_row_pitch = DeviceRowPitch;
  MImpl->MImageFormat = PiFormat;
  MImpl->MImageCopyFlags = detail::getPiImageCopyFlags(
      get_pointer_type(Src, MQueue->get_context()),
      get_pointer_type(Dest, MQueue->get_context()));
  setType(detail::CG::CopyImage);
}

void handler::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MImpl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  setType(detail::CG::SemaphoreWait);
}

void handler::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MImpl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  setType(detail::CG::SemaphoreSignal);
}

void handler::use_kernel_bundle(
    const kernel_bundle<bundle_state::executable> &ExecBundle) {
  std::shared_ptr<detail::queue_impl> PrimaryQueue =
      MImpl->MSubmissionPrimaryQueue;
  if ((!MGraph && (PrimaryQueue->get_context() != ExecBundle.get_context())) ||
      (MGraph && (MGraph->getContext() != ExecBundle.get_context())))
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the primary queue is different from the "
        "context associated with the kernel bundle");

  std::shared_ptr<detail::queue_impl> SecondaryQueue =
      MImpl->MSubmissionSecondaryQueue;
  if (SecondaryQueue &&
      SecondaryQueue->get_context() != ExecBundle.get_context())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the secondary queue is different from the "
        "context associated with the kernel bundle");

  setStateExplicitKernelBundle();
  setHandlerKernelBundle(detail::getSyclObjImpl(ExecBundle));
}

void handler::depends_on(event Event) {
  auto EventImpl = detail::getSyclObjImpl(Event);
  if (EventImpl->isDiscarded()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Queue operation cannot depend on discarded event.");
  }
  if (auto Graph = getCommandGraph(); Graph) {
    auto EventGraph = EventImpl->getCommandGraph();
    if (EventGraph == nullptr) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from outside the graph.");
    }
    if (EventGraph != Graph) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from another graph.");
    }
  }
  CGData.MEvents.push_back(EventImpl);
}

void handler::depends_on(const std::vector<event> &Events) {
  for (const event &Event : Events) {
    depends_on(Event);
  }
}

static bool
checkContextSupports(const std::shared_ptr<detail::context_impl> &ContextImpl,
                     sycl::detail::pi::PiContextInfo InfoQuery) {
  auto &Plugin = ContextImpl->getPlugin();
  pi_bool SupportsOp = false;
  Plugin->call<detail::PiApiKind::piContextGetInfo>(ContextImpl->getHandleRef(),
                                                    InfoQuery, sizeof(pi_bool),
                                                    &SupportsOp, nullptr);
  return SupportsOp;
}

bool handler::supportsUSMMemcpy2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMFill2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMMemset2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT))
      return false;
  }
  return true;
}

id<2> handler::computeFallbackKernelBounds(size_t Width, size_t Height) {
  device Dev = MQueue->get_device();
  range<2> ItemLimit = Dev.get_info<info::device::max_work_item_sizes<2>>() *
                       Dev.get_info<info::device::max_compute_units>();
  return id<2>{std::min(ItemLimit[0], Height), std::min(ItemLimit[1], Width)};
}

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
void handler::ext_intel_read_host_pipe(const detail::string_view Name,
                                       void *Ptr, size_t Size, bool Block) {
  MImpl->HostPipeName = Name.data();
#else
void handler::ext_intel_read_host_pipe(const std::string &Name, void *Ptr,
                                       size_t Size, bool Block) {
  MImpl->HostPipeName = Name;
#endif
  MImpl->HostPipePtr = Ptr;
  MImpl->HostPipeTypeSize = Size;
  MImpl->HostPipeBlocking = Block;
  MImpl->HostPipeRead = 1;
  setType(detail::CG::ReadWriteHostPipe);
}

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
void handler::ext_intel_write_host_pipe(const detail::string_view Name,
                                        void *Ptr, size_t Size, bool Block) {
  MImpl->HostPipeName = Name.data();
#else
void handler::ext_intel_write_host_pipe(const std::string &Name, void *Ptr,
                                        size_t Size, bool Block) {
  MImpl->HostPipeName = Name;
#endif
  MImpl->HostPipePtr = Ptr;
  MImpl->HostPipeTypeSize = Size;
  MImpl->HostPipeBlocking = Block;
  MImpl->HostPipeRead = 0;
  setType(detail::CG::ReadWriteHostPipe);
}

void handler::memcpyToDeviceGlobal(const void *DeviceGlobalPtr, const void *Src,
                                   bool IsDeviceImageScoped, size_t NumBytes,
                                   size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = const_cast<void *>(DeviceGlobalPtr);
  MImpl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  MImpl->MOffset = Offset;
  setType(detail::CG::CopyToDeviceGlobal);
}

void handler::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                     bool IsDeviceImageScoped, size_t NumBytes,
                                     size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(DeviceGlobalPtr);
  MDstPtr = Dest;
  MImpl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  MImpl->MOffset = Offset;
  setType(detail::CG::CopyFromDeviceGlobal);
}

void handler::memcpyToHostOnlyDeviceGlobal(const void *DeviceGlobalPtr,
                                           const void *Src,
                                           size_t DeviceGlobalTSize,
                                           bool IsDeviceImageScoped,
                                           size_t NumBytes, size_t Offset) {
  std::weak_ptr<detail::context_impl> WeakContextImpl =
      MQueue->getContextImplPtr();
  std::weak_ptr<detail::device_impl> WeakDeviceImpl =
      MQueue->getDeviceImplPtr();
  host_task([=] {
    // Capture context and device as weak to avoid keeping them alive for too
    // long. If they are dead by the time this executes, the operation would not
    // have been visible anyway.
    std::shared_ptr<detail::context_impl> ContextImpl = WeakContextImpl.lock();
    std::shared_ptr<detail::device_impl> DeviceImpl = WeakDeviceImpl.lock();
    if (ContextImpl && DeviceImpl)
      ContextImpl->memcpyToHostOnlyDeviceGlobal(
          DeviceImpl, DeviceGlobalPtr, Src, DeviceGlobalTSize,
          IsDeviceImageScoped, NumBytes, Offset);
  });
}

void handler::memcpyFromHostOnlyDeviceGlobal(void *Dest,
                                             const void *DeviceGlobalPtr,
                                             bool IsDeviceImageScoped,
                                             size_t NumBytes, size_t Offset) {
  const std::shared_ptr<detail::context_impl> &ContextImpl =
      MQueue->getContextImplPtr();
  const std::shared_ptr<detail::device_impl> &DeviceImpl =
      MQueue->getDeviceImplPtr();
  host_task([=] {
    // Unlike memcpy to device_global, we need to keep the context and device
    // alive in the capture of this operation as we must be able to correctly
    // copy the value to the user-specified pointer.
    ContextImpl->memcpyFromHostOnlyDeviceGlobal(
        DeviceImpl, Dest, DeviceGlobalPtr, IsDeviceImageScoped, NumBytes,
        Offset);
  });
}

const std::shared_ptr<detail::context_impl> &
handler::getContextImplPtr() const {
  return MQueue->getContextImplPtr();
}

void handler::setKernelCacheConfig(
    sycl::detail::pi::PiKernelCacheConfig Config) {
  MImpl->MKernelCacheConfig = Config;
}

void handler::ext_oneapi_graph(
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::executable>
        Graph) {
  MCGType = detail::CG::ExecCommandBuffer;
  auto GraphImpl = detail::getSyclObjImpl(Graph);
  // GraphImpl is only read in this scope so we lock this graph for read only
  ext::oneapi::experimental::detail::graph_impl::ReadLock Lock(
      GraphImpl->MMutex);

  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> ParentGraph;
  if (MQueue) {
    ParentGraph = MQueue->getCommandGraph();
  } else {
    ParentGraph = MGraph;
  }

  ext::oneapi::experimental::detail::graph_impl::WriteLock ParentLock;
  // If a parent graph is set that means we are adding or recording a subgraph
  if (ParentGraph) {
    // ParentGraph is read and written in this scope so we lock this graph
    // with full priviledges.
    // We only lock for Record&Replay API because the graph has already been
    // lock if this function was called from the explicit API function add
    if (MQueue) {
      ParentLock = ext::oneapi::experimental::detail::graph_impl::WriteLock(
          ParentGraph->MMutex);
    }
    // Store the node representing the subgraph in the handler so that we can
    // return it to the user later.
    // The nodes of the subgraph are duplicated when added to its parents.
    // This avoids changing properties of the graph added as a subgraph.
    MSubgraphNode = ParentGraph->addSubgraphNodes(GraphImpl);

    // If we are recording an in-order queue remember the subgraph node, so it
    // can be used as a dependency for any more nodes recorded from this queue.
    if (MQueue && MQueue->isInOrder()) {
      ParentGraph->setLastInorderNode(MQueue, MSubgraphNode);
    }
    // Associate an event with the subgraph node.
    auto SubgraphEvent = std::make_shared<event_impl>();
    SubgraphEvent->setCommandGraph(ParentGraph);
    ParentGraph->addEventForNode(SubgraphEvent, MSubgraphNode);
  } else {
    // Set the exec graph for execution during finalize.
    MExecGraph = GraphImpl;
  }
}

std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
handler::getCommandGraph() const {
  if (MGraph) {
    return MGraph;
  }
  return MQueue->getCommandGraph();
}

std::optional<std::array<size_t, 3>> handler::getMaxWorkGroups() {
  auto Dev = detail::getSyclObjImpl(detail::getDeviceFromHandler(*this));
  std::array<size_t, 3> PiResult = {};
  auto Ret = Dev->getPlugin()->call_nocheck<PiApiKind::piDeviceGetInfo>(
      Dev->getHandleRef(),
      PiInfoCode<
          ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
      sizeof(PiResult), &PiResult, nullptr);
  if (Ret == PI_SUCCESS) {
    return PiResult;
  }
  return {};
}

} // namespace _V1
} // namespace sycl
