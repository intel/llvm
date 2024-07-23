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
#include <detail/host_task.hpp>
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

#include <sycl/ext/oneapi/bindless_images_memory.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>

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

void *getValueFromDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_base
        &DynamicParamBase) {
  return sycl::detail::getSyclObjImpl(DynamicParamBase)->getValue();
}

} // namespace detail

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 bool CallerNeedsEvent)
    : handler(Queue, Queue, nullptr, CallerNeedsEvent) {}

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 std::shared_ptr<detail::queue_impl> PrimaryQueue,
                 std::shared_ptr<detail::queue_impl> SecondaryQueue,
                 bool CallerNeedsEvent)
    : impl(std::make_shared<detail::handler_impl>(std::move(PrimaryQueue),
                                                  std::move(SecondaryQueue),
                                                  CallerNeedsEvent)),
      MQueue(std::move(Queue)) {}

handler::handler(
    std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph)
    : impl(std::make_shared<detail::handler_impl>(Graph)) {}

// Sets the submission state to indicate that an explicit kernel bundle has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that a specialization constant has been set.
void handler::setStateExplicitKernelBundle() {
  impl->setStateExplicitKernelBundle();
}

// Sets the submission state to indicate that a specialization constant has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that an explicit kernel bundle has been set.
void handler::setStateSpecConstSet() { impl->setStateSpecConstSet(); }

// Returns true if the submission state is EXPLICIT_KERNEL_BUNDLE_STATE and
// false otherwise.
bool handler::isStateExplicitKernelBundle() const {
  return impl->isStateExplicitKernelBundle();
}

// Returns a shared_ptr to the kernel_bundle.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns shared_ptr(nullptr) if Insert is false
std::shared_ptr<detail::kernel_bundle_impl>
handler::getOrInsertHandlerKernelBundle(bool Insert) const {
  if (!impl->MKernelBundle && Insert) {
    auto Ctx =
        impl->MGraph ? impl->MGraph->getContext() : MQueue->get_context();
    auto Dev = impl->MGraph ? impl->MGraph->getDevice() : MQueue->get_device();
    impl->MKernelBundle = detail::getSyclObjImpl(
        get_kernel_bundle<bundle_state::input>(Ctx, {Dev}, {}));
  }
  return impl->MKernelBundle;
}

// Sets kernel bundle to the provided one.
void handler::setHandlerKernelBundle(
    const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr) {
  impl->MKernelBundle = NewKernelBundleImpPtr;
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

  // According to 4.7.6.9 of SYCL2020 spec, if a placeholder accessor is passed
  // to a command without being bound to a command group, an exception should
  // be thrown.
  {
    for (const auto &arg : impl->MArgs) {
      if (arg.MType != detail::kernel_param_kind_t::kind_accessor)
        continue;

      detail::Requirement *AccImpl =
          static_cast<detail::Requirement *>(arg.MPtr);
      if (AccImpl->MIsPlaceH) {
        auto It = std::find(impl->CGData.MRequirements.begin(),
                            impl->CGData.MRequirements.end(), AccImpl);
        if (It == impl->CGData.MRequirements.end())
          throw sycl::exception(make_error_code(errc::kernel_argument),
                                "placeholder accessor must be bound by calling "
                                "handler::require() before it can be used.");

        // Check associated accessors
        bool AccFound = false;
        for (detail::ArgDesc &Acc : impl->MAssociatedAccesors) {
          if (Acc.MType == detail::kernel_param_kind_t::kind_accessor &&
              static_cast<detail::Requirement *>(Acc.MPtr) == AccImpl) {
            AccFound = true;
            break;
          }
        }

        if (!AccFound) {
          throw sycl::exception(make_error_code(errc::kernel_argument),
                                "placeholder accessor must be bound by calling "
                                "handler::require() before it can be used.");
        }
      }
    }
  }

  const auto &type = getType();
  if (type == detail::CGType::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/false);
    if (KernelBundleImpPtr) {
      // Make sure implicit non-interop kernel bundles have the kernel
      if (!KernelBundleImpPtr->isInterop() &&
          !impl->isStateExplicitKernelBundle()) {
        auto Dev =
            impl->MGraph ? impl->MGraph->getDevice() : MQueue->get_device();
        kernel_id KernelID =
            detail::ProgramManager::getInstance().getSYCLKernelID(
                MKernelName.c_str());
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

    if (MQueue && !impl->MGraph && !impl->MSubgraphNode &&
        !MQueue->getCommandGraph() && !MQueue->is_in_fusion_mode() &&
        !impl->CGData.MRequirements.size() && !MStreamStorage.size() &&
        (!impl->CGData.MEvents.size() ||
         (MQueue->isInOrder() &&
          detail::Scheduler::areEventsSafeForSchedulerBypass(
              impl->CGData.MEvents, MQueue->getContextImplPtr())))) {
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
          StreamID, MKernel, MCodeLoc, MKernelName.c_str(), MQueue,
          impl->MNDRDesc, KernelBundleImpPtr, impl->MArgs);
      auto EnqueueKernel = [&, CmdTraceEvent = CmdTraceEvent,
                            InstanceID = InstanceID]() {
#else
      auto EnqueueKernel = [&]() {
#endif
#ifdef XPTI_ENABLE_INSTRUMENTATION
        detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                           xpti::trace_task_begin, nullptr);
#endif
        enqueueImpKernel(MQueue, impl->MNDRDesc, impl->MArgs,
                         KernelBundleImpPtr, MKernel, MKernelName.c_str(),
                         RawEvents, NewEvent, nullptr, impl->MKernelCacheConfig,
                         impl->MKernelIsCooperative,
                         impl->MKernelUsesClusterLaunch);
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
      };

      bool DiscardEvent = (MQueue->MDiscardEvents || !impl->MEventNeeded) &&
                          MQueue->supportsDiscardingPiEvents();
      if (DiscardEvent) {
        // Kernel only uses assert if it's non interop one
        bool KernelUsesAssert =
            !(MKernel && MKernel->isInterop()) &&
            detail::ProgramManager::getInstance().kernelUsesAssert(
                MKernelName.c_str());
        DiscardEvent = !KernelUsesAssert;
      }

      if (DiscardEvent) {
        EnqueueKernel();
        auto EventImpl = std::make_shared<detail::event_impl>(
            detail::event_impl::HES_Discarded);
        MLastEvent = detail::createSyclObjFromImpl<event>(EventImpl);
      } else {
        NewEvent = std::make_shared<detail::event_impl>(MQueue);
        NewEvent->setWorkerQueue(MQueue);
        NewEvent->setContextImpl(MQueue->getContextImplPtr());
        NewEvent->setStateIncomplete();
        NewEvent->setSubmissionTime();

        EnqueueKernel();
        if (NewEvent->isHost() || NewEvent->getHandleRef() == nullptr)
          NewEvent->setComplete();
        NewEvent->setEnqueued();

        MLastEvent = detail::createSyclObjFromImpl<event>(NewEvent);
      }
      return MLastEvent;
    }
  }

  std::unique_ptr<detail::CG> CommandGroup;
  switch (type) {
  case detail::CGType::Kernel: {
    // Copy kernel name here instead of move so that it's available after
    // running of this method by reductions implementation. This allows for
    // assert feature to check if kernel uses assertions
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(impl->MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(impl->MKernelBundle), std::move(impl->CGData),
        std::move(impl->MArgs), MKernelName.c_str(), std::move(MStreamStorage),
        std::move(impl->MAuxiliaryResources), getType(),
        impl->MKernelCacheConfig, impl->MKernelIsCooperative,
        impl->MKernelUsesClusterLaunch, MCodeLoc));
    break;
  }
  case detail::CGType::CopyAccToPtr:
  case detail::CGType::CopyPtrToAcc:
  case detail::CGType::CopyAccToAcc:
    CommandGroup.reset(
        new detail::CGCopy(getType(), MSrcPtr, MDstPtr, std::move(impl->CGData),
                           std::move(impl->MAuxiliaryResources), MCodeLoc));
    break;
  case detail::CGType::Fill:
    CommandGroup.reset(new detail::CGFill(std::move(MPattern), MDstPtr,
                                          std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::UpdateHost:
    CommandGroup.reset(
        new detail::CGUpdateHost(MDstPtr, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::CopyUSM:
    CommandGroup.reset(new detail::CGCopyUSM(
        MSrcPtr, MDstPtr, MLength, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::FillUSM:
    CommandGroup.reset(new detail::CGFillUSM(std::move(MPattern), MDstPtr,
                                             MLength, std::move(impl->CGData),
                                             MCodeLoc));
    break;
  case detail::CGType::PrefetchUSM:
    CommandGroup.reset(new detail::CGPrefetchUSM(
        MDstPtr, MLength, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::AdviseUSM:
    CommandGroup.reset(new detail::CGAdviseUSM(MDstPtr, MLength, impl->MAdvice,
                                               std::move(impl->CGData),
                                               getType(), MCodeLoc));
    break;
  case detail::CGType::Copy2DUSM:
    CommandGroup.reset(new detail::CGCopy2DUSM(
        MSrcPtr, MDstPtr, impl->MSrcPitch, impl->MDstPitch, impl->MWidth,
        impl->MHeight, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::Fill2DUSM:
    CommandGroup.reset(new detail::CGFill2DUSM(
        std::move(MPattern), MDstPtr, impl->MDstPitch, impl->MWidth,
        impl->MHeight, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::Memset2DUSM:
    CommandGroup.reset(new detail::CGMemset2DUSM(
        MPattern[0], MDstPtr, impl->MDstPitch, impl->MWidth, impl->MHeight,
        std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::EnqueueNativeCommand:
  case detail::CGType::CodeplayHostTask: {
    auto context = impl->MGraph
                       ? detail::getSyclObjImpl(impl->MGraph->getContext())
                       : MQueue->getContextImplPtr();
    CommandGroup.reset(new detail::CGHostTask(
        std::move(impl->MHostTask), MQueue, context, std::move(impl->MArgs),
        std::move(impl->CGData), getType(), MCodeLoc));
    break;
  }
  case detail::CGType::Barrier:
  case detail::CGType::BarrierWaitlist: {
    if (auto GraphImpl = getCommandGraph(); GraphImpl != nullptr) {
      impl->CGData.MEvents.insert(std::end(impl->CGData.MEvents),
                                  std::begin(impl->MEventsWaitWithBarrier),
                                  std::end(impl->MEventsWaitWithBarrier));
      // Barrier node is implemented as an empty node in Graph
      // but keep the barrier type to help managing dependencies
      setType(detail::CGType::Barrier);
      CommandGroup.reset(new detail::CG(detail::CGType::Barrier,
                                        std::move(impl->CGData), MCodeLoc));
    } else {
      CommandGroup.reset(
          new detail::CGBarrier(std::move(impl->MEventsWaitWithBarrier),
                                std::move(impl->CGData), getType(), MCodeLoc));
    }
    break;
  }
  case detail::CGType::ProfilingTag: {
    CommandGroup.reset(
        new detail::CGProfilingTag(std::move(impl->CGData), MCodeLoc));
    break;
  }
  case detail::CGType::CopyToDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyToDeviceGlobal(
        MSrcPtr, MDstPtr, impl->MIsDeviceImageScoped, MLength, impl->MOffset,
        std::move(impl->CGData), MCodeLoc));
    break;
  }
  case detail::CGType::CopyFromDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyFromDeviceGlobal(
        MSrcPtr, MDstPtr, impl->MIsDeviceImageScoped, MLength, impl->MOffset,
        std::move(impl->CGData), MCodeLoc));
    break;
  }
  case detail::CGType::ReadWriteHostPipe: {
    CommandGroup.reset(new detail::CGReadWriteHostPipe(
        impl->HostPipeName, impl->HostPipeBlocking, impl->HostPipePtr,
        impl->HostPipeTypeSize, impl->HostPipeRead, std::move(impl->CGData),
        MCodeLoc));
    break;
  }
  case detail::CGType::ExecCommandBuffer: {
    std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> ParentGraph =
        MQueue ? MQueue->getCommandGraph() : impl->MGraph;

    // If a parent graph is set that means we are adding or recording a subgraph
    // and we don't want to actually execute this command graph submission.
    if (ParentGraph) {
      ext::oneapi::experimental::detail::graph_impl::WriteLock ParentLock;
      if (MQueue) {
        ParentLock = ext::oneapi::experimental::detail::graph_impl::WriteLock(
            ParentGraph->MMutex);
      }
      impl->CGData.MRequirements = impl->MExecGraph->getRequirements();
      // Here we are using the CommandGroup without passing a CommandBuffer to
      // pass the exec_graph_impl and event dependencies. Since this subgraph CG
      // will not be executed this is fine.
      CommandGroup.reset(new sycl::detail::CGExecCommandBuffer(
          nullptr, impl->MExecGraph, std::move(impl->CGData)));

    } else {
      event GraphCompletionEvent =
          impl->MExecGraph->enqueue(MQueue, std::move(impl->CGData));
      MLastEvent = GraphCompletionEvent;
      return MLastEvent;
    }
  } break;
  case detail::CGType::CopyImage:
    CommandGroup.reset(new detail::CGCopyImage(
        MSrcPtr, MDstPtr, impl->MSrcImageDesc, impl->MDestImageDesc,
        impl->MSrcImageFormat, impl->MDestImageFormat, impl->MImageCopyFlags,
        impl->MSrcOffset, impl->MDestOffset, impl->MCopyExtent,
        std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::SemaphoreWait:
    CommandGroup.reset(new detail::CGSemaphoreWait(
        impl->MInteropSemaphoreHandle, impl->MWaitValue,
        std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::SemaphoreSignal:
    CommandGroup.reset(new detail::CGSemaphoreSignal(
        impl->MInteropSemaphoreHandle, impl->MSignalValue,
        std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::None:
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      std::cout << "WARNING: An empty command group is submitted." << std::endl;
    }

    // Empty nodes are handled by Graph like standard nodes
    // For Standard mode (non-graph),
    // empty nodes are not sent to the scheduler to save time
    if (impl->MGraph || (MQueue && MQueue->getCommandGraph())) {
      CommandGroup.reset(new detail::CG(detail::CGType::None,
                                        std::move(impl->CGData), MCodeLoc));
    } else {
      detail::EventImplPtr Event = std::make_shared<sycl::detail::event_impl>();
      MLastEvent = detail::createSyclObjFromImpl<event>(Event);
      return MLastEvent;
    }
    break;
  }

  if (!CommandGroup)
    throw exception(make_error_code(errc::runtime),
                    "Internal Error. Command group cannot be constructed.");

  // If there is a graph associated with the handler we are in the explicit
  // graph mode, so we store the CG instead of submitting it to the scheduler,
  // so it can be retrieved by the graph later.
  if (impl->MGraph) {
    impl->MGraphNodeCG = std::move(CommandGroup);
    return detail::createSyclObjFromImpl<event>(
        std::make_shared<detail::event_impl>());
  }

  // If the queue has an associated graph then we need to take the CG and pass
  // it to the graph to create a node, rather than submit it to the scheduler.
  if (auto GraphImpl = MQueue->getCommandGraph(); GraphImpl) {
    auto EventImpl = std::make_shared<detail::event_impl>();
    EventImpl->setSubmittedQueue(MQueue);
    std::shared_ptr<ext::oneapi::experimental::detail::node_impl> NodeImpl =
        nullptr;

    // GraphImpl is read and written in this scope so we lock this graph
    // with full priviledges.
    ext::oneapi::experimental::detail::graph_impl::WriteLock Lock(
        GraphImpl->MMutex);

    ext::oneapi::experimental::node_type NodeType =
        impl->MUserFacingNodeType != ext::oneapi::experimental::node_type::empty
            ? impl->MUserFacingNodeType
            : ext::oneapi::experimental::detail::getNodeTypeFromCG(getType());

    // Create a new node in the graph representing this command-group
    if (MQueue->isInOrder()) {
      // In-order queues create implicit linear dependencies between nodes.
      // Find the last node added to the graph from this queue, so our new
      // node can set it as a predecessor.
      auto DependentNode = GraphImpl->getLastInorderNode(MQueue);

      NodeImpl = DependentNode
                     ? GraphImpl->add(NodeType, std::move(CommandGroup),
                                      {DependentNode})
                     : GraphImpl->add(NodeType, std::move(CommandGroup));

      // If we are recording an in-order queue remember the new node, so it
      // can be used as a dependency for any more nodes recorded from this
      // queue.
      GraphImpl->setLastInorderNode(MQueue, NodeImpl);
    } else {
      auto LastBarrierRecordedFromQueue = GraphImpl->getBarrierDep(MQueue);
      if (LastBarrierRecordedFromQueue) {
        NodeImpl = GraphImpl->add(NodeType, std::move(CommandGroup),
                                  {LastBarrierRecordedFromQueue});
      } else {
        NodeImpl = GraphImpl->add(NodeType, std::move(CommandGroup));
      }

      if (NodeImpl->MCGType == sycl::detail::CGType::Barrier) {
        GraphImpl->setBarrierDep(MQueue, NodeImpl);
      }
    }

    // Associate an event with this new node and return the event.
    GraphImpl->addEventForNode(GraphImpl, EventImpl, NodeImpl);

    NodeImpl->MNDRangeUsed = impl->MNDRangeUsed;

    return detail::createSyclObjFromImpl<event>(EventImpl);
  }

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue), impl->MEventNeeded);

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);
  return MLastEvent;
}

void handler::addReduction(const std::shared_ptr<const void> &ReduObj) {
  impl->MAuxiliaryResources.push_back(ReduObj);
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
    impl->CGData.MRequirements.push_back(Req);
  // Store copy of the accessor.
  impl->CGData.MAccStorage.push_back(std::move(AccImpl));
  // Add an accessor to the handler list of associated accessors.
  // For associated accessors index does not means nothing.
  impl->MAssociatedAccesors.emplace_back(
      detail::kernel_param_kind_t::kind_accessor, Req, AccTarget, /*index*/ 0);
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
    addArg(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_stream: {
    // Stream contains several accessors inside.
    stream *S = static_cast<stream *>(Ptr);

    detail::AccessorBaseHost *GBufBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalBuf);
    detail::AccessorImplPtr GBufImpl = detail::getSyclObjImpl(*GBufBase);
    detail::Requirement *GBufReq = GBufImpl.get();
    addArgsForGlobalAccessor(
        GBufReq, Index, IndexShift, Size, IsKernelCreatedFromSource,
        impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GOffsetBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalOffset);
    detail::AccessorImplPtr GOfssetImpl = detail::getSyclObjImpl(*GOffsetBase);
    detail::Requirement *GOffsetReq = GOfssetImpl.get();
    addArgsForGlobalAccessor(
        GOffsetReq, Index, IndexShift, Size, IsKernelCreatedFromSource,
        impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GFlushBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalFlushBuf);
    detail::AccessorImplPtr GFlushImpl = detail::getSyclObjImpl(*GFlushBase);
    detail::Requirement *GFlushReq = GFlushImpl.get();

    size_t GlobalSize = impl->MNDRDesc.GlobalSize.size();
    // If work group size wasn't set explicitly then it must be recieved
    // from kernel attribute or set to default values.
    // For now we can't get this attribute here.
    // So we just suppose that WG size is always default for stream.
    // TODO adjust MNDRDesc when device image contains kernel's attribute
    if (GlobalSize == 0) {
      // Suppose that work group size is 1 for every dimension
      GlobalSize = impl->MNDRDesc.NumWorkGroups.size();
    }
    addArgsForGlobalAccessor(GFlushReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, impl->MArgs,
                             IsESIMD);
    ++IndexShift;
    addArg(kernel_param_kind_t::kind_std_layout, &S->FlushBufferSize,
           sizeof(S->FlushBufferSize), Index + IndexShift);

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
      addArgsForGlobalAccessor(
          AccImpl, Index, IndexShift, Size, IsKernelCreatedFromSource,
          impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
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
      impl->MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                               SizeInBytes, Index + IndexShift);
      // TODO ESIMD currently does not suport MSize field passing yet
      // accessor::init for ESIMD-mode accessor has a single field, translated
      // to a single kernel argument set above.
      if (!IsESIMD && !IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = (Dims == 0 ? 1 : Dims) * sizeof(Size[0]);
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
               Index + IndexShift);
        ++IndexShift;
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
               Index + IndexShift);
        ++IndexShift;
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
               Index + IndexShift);
      }
      break;
    }
    case access::target::image:
    case access::target::image_array: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      addArg(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // TODO Handle additional kernel arguments for image class
        // if the compiler front-end adds them.
      }
      break;
    }
    case access::target::host_image:
    case access::target::host_task:
    case access::target::host_buffer: {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Unsupported accessor target case.");
      break;
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    addArg(kernel_param_kind_t::kind_sampler, Ptr, sizeof(sampler),
           Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    addArg(kernel_param_kind_t::kind_specialization_constants_buffer, Ptr, Size,
           Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw exception(make_error_code(errc::invalid),
                    "Invalid kernel param kind");
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
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(impl->MArgs);
  clearArgs();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  const bool IsKernelCreatedFromSource = MKernel->isCreatedFromSource();
  impl->MArgs.reserve(MaxNumAdditionalArgs * UnPreparedArgs.size());

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
  impl->MArgs.reserve(MaxNumAdditionalArgs * KernelArgsNum);

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
detail::string handler::getKernelName() {
  return detail::string{MKernel->get_info<info::kernel::function_name>()};
}

void handler::verifyUsedKernelBundleInternal(detail::string_view KernelName) {
  auto UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundle(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  if (!impl->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  device Dev = impl->MGraph ? impl->MGraph->getDevice()
                            : detail::getDeviceFromHandler(*this);
  if (!UsedKernelBundleImplPtr->has_kernel(KernelID, Dev))
    throw sycl::exception(
        make_error_code(errc::kernel_not_supported),
        "The kernel bundle in use does not contain the kernel");
}

void handler::ext_oneapi_barrier(const std::vector<event> &WaitList) {
  throwIfActionIsCreated();
  setType(detail::CGType::BarrierWaitlist);
  impl->MEventsWaitWithBarrier.reserve(WaitList.size());
  for (auto &Event : WaitList) {
    auto EventImpl = detail::getSyclObjImpl(Event);
    // We could not wait for host task events in backend.
    // Adding them as dependency to enable proper scheduling.
    if (EventImpl->isHost()) {
      depends_on(EventImpl);
    }
    impl->MEventsWaitWithBarrier.push_back(EventImpl);
  }
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
  setType(detail::CGType::CopyUSM);
}

void handler::memset(void *Dest, int Value, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MLength = Count;
  setUserFacingNodeType(ext::oneapi::experimental::node_type::memset);
  setType(detail::CGType::FillUSM);
}

void handler::prefetch(const void *Ptr, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  setType(detail::CGType::PrefetchUSM);
}

void handler::mem_advise(const void *Ptr, size_t Count, int Advice) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  impl->MAdvice = static_cast<pi_mem_advice>(Advice);
  setType(detail::CGType::AdviseUSM);
}

void handler::fill_impl(void *Dest, const void *Value, size_t ValueSize,
                        size_t Count) {
  MDstPtr = Dest;
  MPattern.resize(ValueSize);
  std::memcpy(MPattern.data(), Value, ValueSize);
  MLength = Count * ValueSize;
  setType(detail::CGType::FillUSM);
}

void handler::ext_oneapi_memcpy2d_impl(void *Dest, size_t DestPitch,
                                       const void *Src, size_t SrcPitch,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  impl->MSrcPitch = SrcPitch;
  impl->MDstPitch = DestPitch;
  impl->MWidth = Width;
  impl->MHeight = Height;
  setType(detail::CGType::Copy2DUSM);
}

void handler::ext_oneapi_fill2d_impl(void *Dest, size_t DestPitch,
                                     const void *Value, size_t ValueSize,
                                     size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.resize(ValueSize);
  std::memcpy(MPattern.data(), Value, ValueSize);
  impl->MDstPitch = DestPitch;
  impl->MWidth = Width;
  impl->MHeight = Height;
  setType(detail::CGType::Fill2DUSM);
}

void handler::ext_oneapi_memset2d_impl(void *Dest, size_t DestPitch, int Value,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.push_back(static_cast<unsigned char>(Value));
  impl->MDstPitch = DestPitch;
  impl->MWidth = Width;
  impl->MHeight = Height;
  setType(detail::CGType::Memset2DUSM);
}

void handler::ext_oneapi_copy(
    const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &Desc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  Desc.verify();

  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest.raw_handle;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_array_size = Desc.array_size;

  if (Desc.array_size > 1) {
    // Image Array.
    PiDesc.image_type =
        Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        Desc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type =
        Desc.depth > 0
            ? PI_MEM_TYPE_IMAGE3D
            : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(Desc.num_channels));

  impl->MSrcOffset = {0, 0, 0};
  impl->MDestOffset = {0, 0, 0};
  impl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE;
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  DestImgDesc.verify();

  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest.raw_handle;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = DestImgDesc.width;
  PiDesc.image_height = DestImgDesc.height;
  PiDesc.image_depth = DestImgDesc.depth;
  PiDesc.image_array_size = DestImgDesc.array_size;

  if (DestImgDesc.array_size > 1) {
    // Image Array.
    PiDesc.image_type = DestImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY
                                               : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        DestImgDesc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type = DestImgDesc.depth > 0
                            ? PI_MEM_TYPE_IMAGE3D
                            : (DestImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                      : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(DestImgDesc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(DestImgDesc.num_channels));

  impl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  impl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  impl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  impl->MSrcImageDesc = PiDesc;
  impl->MSrcImageDesc.image_width = SrcExtent[0];
  impl->MSrcImageDesc.image_height = SrcExtent[1];
  impl->MSrcImageDesc.image_depth = SrcExtent[2];
  impl->MDestImageDesc = PiDesc;
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE;
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &Desc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  Desc.verify();

  MSrcPtr = Src.raw_handle;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_array_size = Desc.array_size;

  if (Desc.array_size > 1) {
    // Image Array.
    PiDesc.image_type =
        Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        Desc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type =
        Desc.depth > 0
            ? PI_MEM_TYPE_IMAGE3D
            : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(Desc.num_channels));

  impl->MSrcOffset = {0, 0, 0};
  impl->MDestOffset = {0, 0, 0};
  impl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST;
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &ImageDesc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  ImageDesc.verify();

  MSrcPtr = Src.raw_handle;
  MDstPtr = Dest.raw_handle;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = ImageDesc.width;
  PiDesc.image_height = ImageDesc.height;
  PiDesc.image_depth = ImageDesc.depth;
  PiDesc.image_array_size = ImageDesc.array_size;
  if (ImageDesc.array_size > 1) {
    // Image Array.
    PiDesc.image_type = ImageDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY
                                             : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        ImageDesc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type = ImageDesc.depth > 0
                            ? PI_MEM_TYPE_IMAGE3D
                            : (ImageDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                    : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(ImageDesc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(ImageDesc.num_channels));

  impl->MSrcOffset = {0, 0, 0};
  impl->MDestOffset = {0, 0, 0};
  impl->MCopyExtent = {ImageDesc.width, ImageDesc.height, ImageDesc.depth};
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_DEVICE;
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  SrcImgDesc.verify();

  MSrcPtr = Src.raw_handle;
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = SrcImgDesc.width;
  PiDesc.image_height = SrcImgDesc.height;
  PiDesc.image_depth = SrcImgDesc.depth;
  PiDesc.image_array_size = SrcImgDesc.array_size;

  if (SrcImgDesc.array_size > 1) {
    // Image Array.
    PiDesc.image_type = SrcImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY
                                              : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        SrcImgDesc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type = SrcImgDesc.depth > 0
                            ? PI_MEM_TYPE_IMAGE3D
                            : (SrcImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                     : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(SrcImgDesc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(SrcImgDesc.num_channels));

  impl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  impl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  impl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;
  impl->MDestImageDesc.image_width = DestExtent[0];
  impl->MDestImageDesc.image_height = DestExtent[1];
  impl->MDestImageDesc.image_depth = DestExtent[2];
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags =
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST;
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &Desc, size_t Pitch) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  Desc.verify();

  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_array_size = Desc.array_size;

  if (Desc.array_size > 1) {
    // Image Array.
    PiDesc.image_type =
        Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        Desc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type =
        Desc.depth > 0
            ? PI_MEM_TYPE_IMAGE3D
            : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(Desc.num_channels));

  impl->MSrcOffset = {0, 0, 0};
  impl->MDestOffset = {0, 0, 0};
  impl->MCopyExtent = {Desc.width, Desc.height, Desc.depth};
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MSrcImageDesc.image_row_pitch = Pitch;
  impl->MDestImageDesc.image_row_pitch = Pitch;
  impl->MImageCopyFlags = detail::getPiImageCopyFlags(
      get_pointer_type(Src, MQueue->get_context()),
      get_pointer_type(Dest, MQueue->get_context()));
  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  DeviceImgDesc.verify();

  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  sycl::detail::pi::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = DeviceImgDesc.width;
  PiDesc.image_height = DeviceImgDesc.height;
  PiDesc.image_depth = DeviceImgDesc.depth;
  PiDesc.image_array_size = DeviceImgDesc.array_size;

  if (DeviceImgDesc.array_size > 1) {
    // Image Array.
    PiDesc.image_type = DeviceImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D_ARRAY
                                                 : PI_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    PiDesc.image_type =
        DeviceImgDesc.type ==
                sycl::ext::oneapi::experimental::image_type::cubemap
            ? PI_MEM_TYPE_IMAGE_CUBEMAP
            : PiDesc.image_type;
  } else {
    PiDesc.image_type = DeviceImgDesc.depth > 0
                            ? PI_MEM_TYPE_IMAGE3D
                            : (DeviceImgDesc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  }

  sycl::detail::pi::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(DeviceImgDesc.channel_type);
  PiFormat.image_channel_order = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(DeviceImgDesc.num_channels));

  impl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  impl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  impl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  impl->MSrcImageFormat = PiFormat;
  impl->MDestImageFormat = PiFormat;
  impl->MImageCopyFlags = detail::getPiImageCopyFlags(
      get_pointer_type(Src, MQueue->get_context()),
      get_pointer_type(Dest, MQueue->get_context()));
  impl->MSrcImageDesc = PiDesc;
  impl->MDestImageDesc = PiDesc;

  // Fill the descriptor row pitch and host extent based on the type of copy.
  if (impl->MImageCopyFlags ==
      sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_HOST_TO_DEVICE) {
    impl->MDestImageDesc.image_row_pitch = DeviceRowPitch;
    impl->MSrcImageDesc.image_row_pitch = 0;
    impl->MSrcImageDesc.image_width = HostExtent[0];
    impl->MSrcImageDesc.image_height = HostExtent[1];
    impl->MSrcImageDesc.image_depth = HostExtent[2];
  } else if (impl->MImageCopyFlags ==
             sycl::detail::pi::PiImageCopyFlags::PI_IMAGE_COPY_DEVICE_TO_HOST) {
    impl->MSrcImageDesc.image_row_pitch = DeviceRowPitch;
    impl->MDestImageDesc.image_row_pitch = 0;
    impl->MDestImageDesc.image_width = HostExtent[0];
    impl->MDestImageDesc.image_height = HostExtent[1];
    impl->MDestImageDesc.image_depth = HostExtent[2];
  } else {
    impl->MDestImageDesc.image_row_pitch = DeviceRowPitch;
    impl->MSrcImageDesc.image_row_pitch = DeviceRowPitch;
  }

  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  if (SemaphoreHandle.handle_type !=
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              opaque_fd &&
      SemaphoreHandle.handle_type !=
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              win32_nt_handle) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore used needs a user passed wait value.");
  }
  impl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  impl->MWaitValue = {};
  setType(detail::CGType::SemaphoreWait);
}

void handler::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t WaitValue) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  if (SemaphoreHandle.handle_type !=
      sycl::ext::oneapi::experimental::external_semaphore_handle_type::
          win32_nt_dx12_fence) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore does not support user passed wait values.");
  }
  impl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  impl->MWaitValue = WaitValue;
  setType(detail::CGType::SemaphoreWait);
}

void handler::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  if (SemaphoreHandle.handle_type !=
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              opaque_fd &&
      SemaphoreHandle.handle_type !=
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              win32_nt_handle) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore used needs a user passed signal value.");
  }
  impl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  impl->MSignalValue = {};
  setType(detail::CGType::SemaphoreSignal);
}

void handler::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t SignalValue) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  if (SemaphoreHandle.handle_type !=
      sycl::ext::oneapi::experimental::external_semaphore_handle_type::
          win32_nt_dx12_fence) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore does not support user passed signal values.");
  }
  impl->MInteropSemaphoreHandle =
      (sycl::detail::pi::PiInteropSemaphoreHandle)SemaphoreHandle.raw_handle;
  impl->MSignalValue = SignalValue;
  setType(detail::CGType::SemaphoreSignal);
}

void handler::use_kernel_bundle(
    const kernel_bundle<bundle_state::executable> &ExecBundle) {
  std::shared_ptr<detail::queue_impl> PrimaryQueue =
      impl->MSubmissionPrimaryQueue;
  if ((!impl->MGraph &&
       (PrimaryQueue->get_context() != ExecBundle.get_context())) ||
      (impl->MGraph &&
       (impl->MGraph->getContext() != ExecBundle.get_context())))
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the primary queue is different from the "
        "context associated with the kernel bundle");

  std::shared_ptr<detail::queue_impl> SecondaryQueue =
      impl->MSubmissionSecondaryQueue;
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
  depends_on(EventImpl);
}

void handler::depends_on(const std::vector<event> &Events) {
  for (const event &Event : Events) {
    depends_on(Event);
  }
}

void handler::depends_on(const detail::EventImplPtr &EventImpl) {
  if (!EventImpl)
    return;
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
  impl->CGData.MEvents.push_back(EventImpl);
}

void handler::depends_on(const std::vector<detail::EventImplPtr> &Events) {
  for (const EventImplPtr &Event : Events) {
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

void handler::verifyDeviceHasProgressGuarantee(
    sycl::ext::oneapi::experimental::forward_progress_guarantee guarantee,
    sycl::ext::oneapi::experimental::execution_scope threadScope,
    sycl::ext::oneapi::experimental::execution_scope coordinationScope) {
  using execution_scope = sycl::ext::oneapi::experimental::execution_scope;
  using forward_progress =
      sycl::ext::oneapi::experimental::forward_progress_guarantee;
  auto deviceImplPtr = MQueue->getDeviceImplPtr();
  const bool supported = deviceImplPtr->supportsForwardProgress(
      guarantee, threadScope, coordinationScope);
  if (threadScope == execution_scope::work_group) {
    if (!supported) {
      throw sycl::exception(
          sycl::errc::feature_not_supported,
          "Required progress guarantee for work groups is not "
          "supported by this device.");
    }
    // If we are here, the device supports the guarantee required but there is a
    // caveat in that if the guarantee required is a concurrent guarantee, then
    // we most likely also need to enable cooperative launch of the kernel. That
    // is, although the device supports the required guarantee, some setup work
    // is needed to truly make the device provide that guarantee at runtime.
    // Otherwise, we will get the default guarantee which is weaker than
    // concurrent. Same reasoning applies for sub_group but not for work_item.
    // TODO: Further design work is probably needed to reflect this behavior in
    // Unified Runtime.
    if (guarantee == forward_progress::concurrent)
      setKernelIsCooperative(true);
  } else if (threadScope == execution_scope::sub_group) {
    if (!supported) {
      throw sycl::exception(sycl::errc::feature_not_supported,
                            "Required progress guarantee for sub groups is not "
                            "supported by this device.");
    }
    // Same reasoning as above.
    if (guarantee == forward_progress::concurrent)
      setKernelIsCooperative(true);
  } else { // threadScope is execution_scope::work_item otherwise undefined
           // behavior
    if (!supported) {
      throw sycl::exception(sycl::errc::feature_not_supported,
                            "Required progress guarantee for work items is not "
                            "supported by this device.");
    }
  }
}

bool handler::supportsUSMMemcpy2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {impl->MSubmissionPrimaryQueue, impl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMFill2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {impl->MSubmissionPrimaryQueue, impl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMMemset2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {impl->MSubmissionPrimaryQueue, impl->MSubmissionSecondaryQueue}) {
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

backend handler::getDeviceBackend() const {
  if (impl->MGraph)
    return impl->MGraph->getDevice().get_backend();
  else
    return MQueue->getDeviceImplPtr()->getBackend();
}

void handler::ext_intel_read_host_pipe(detail::string_view Name, void *Ptr,
                                       size_t Size, bool Block) {
  impl->HostPipeName = Name.data();
  impl->HostPipePtr = Ptr;
  impl->HostPipeTypeSize = Size;
  impl->HostPipeBlocking = Block;
  impl->HostPipeRead = 1;
  setType(detail::CGType::ReadWriteHostPipe);
}

void handler::ext_intel_write_host_pipe(detail::string_view Name, void *Ptr,
                                        size_t Size, bool Block) {
  impl->HostPipeName = Name.data();
  impl->HostPipePtr = Ptr;
  impl->HostPipeTypeSize = Size;
  impl->HostPipeBlocking = Block;
  impl->HostPipeRead = 0;
  setType(detail::CGType::ReadWriteHostPipe);
}

void handler::memcpyToDeviceGlobal(const void *DeviceGlobalPtr, const void *Src,
                                   bool IsDeviceImageScoped, size_t NumBytes,
                                   size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = const_cast<void *>(DeviceGlobalPtr);
  impl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  impl->MOffset = Offset;
  setType(detail::CGType::CopyToDeviceGlobal);
}

void handler::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                     bool IsDeviceImageScoped, size_t NumBytes,
                                     size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(DeviceGlobalPtr);
  MDstPtr = Dest;
  impl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  impl->MOffset = Offset;
  setType(detail::CGType::CopyFromDeviceGlobal);
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

void handler::setKernelCacheConfig(handler::StableKernelCacheConfig Config) {
  switch (Config) {
    case handler::StableKernelCacheConfig::Default:
      impl->MKernelCacheConfig = PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT;
      break;
    case handler::StableKernelCacheConfig::LargeSLM:
      impl->MKernelCacheConfig = PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM;
      break;
    case handler::StableKernelCacheConfig::LargeData:
      impl->MKernelCacheConfig = PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA;
      break;
  }
}

void handler::setKernelIsCooperative(bool KernelIsCooperative) {
  impl->MKernelIsCooperative = KernelIsCooperative;
}

void handler::setKernelClusterLaunch(sycl::range<3> ClusterSize, int Dims) {
  throwIfGraphAssociated<
      syclex::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_experimental_cuda_cluster_launch>();
  impl->MKernelUsesClusterLaunch = true;
  impl->MNDRDesc.setClusterDimensions(ClusterSize, Dims);
}

void handler::ext_oneapi_graph(
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::executable>
        Graph) {
  setType(detail::CGType::ExecCommandBuffer);
  impl->MExecGraph = detail::getSyclObjImpl(Graph);
}

std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
handler::getCommandGraph() const {
  if (impl->MGraph) {
    return impl->MGraph;
  }
  return MQueue->getCommandGraph();
}

void handler::setUserFacingNodeType(ext::oneapi::experimental::node_type Type) {
  impl->MUserFacingNodeType = Type;
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

std::tuple<std::array<size_t, 3>, bool> handler::getMaxWorkGroups_v2() {
  auto ImmRess = getMaxWorkGroups();
  if (ImmRess)
    return {*ImmRess, true};
  return {std::array<size_t, 3>{0, 0, 0}, false};
}

void handler::setNDRangeUsed(bool Value) { impl->MNDRangeUsed = Value; }

void handler::registerDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_base &DynamicParamBase,
    int ArgIndex) {
  if (MQueue && MQueue->getCommandGraph()) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Dynamic Parameters cannot be used with Graph Queue recording.");
  }
  if (!impl->MGraph) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Dynamic Parameters cannot be used with normal SYCL submissions");
  }

  auto Paraimpl = detail::getSyclObjImpl(DynamicParamBase);
  if (Paraimpl->MGraph != this->impl->MGraph) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Cannot use a Dynamic Parameter with a node associated with a graph "
        "other than the one it was created with.");
  }
  impl->MDynamicParameters.emplace_back(Paraimpl.get(), ArgIndex);
}

bool handler::eventNeeded() const { return impl->MEventNeeded; }

void *handler::storeRawArg(const void *Ptr, size_t Size) {
  impl->CGData.MArgsStorage.emplace_back(Size);
  void *Storage = static_cast<void *>(impl->CGData.MArgsStorage.back().data());
  std::memcpy(Storage, Ptr, Size);
  return Storage;
}

void handler::SetHostTask(std::function<void()> &&Func) {
  setNDRangeDescriptor(range<1>(1));
  impl->MHostTask.reset(new detail::HostTask(std::move(Func)));
  setType(detail::CGType::CodeplayHostTask);
}

void handler::SetHostTask(std::function<void(interop_handle)> &&Func) {
  setNDRangeDescriptor(range<1>(1));
  impl->MHostTask.reset(new detail::HostTask(std::move(Func)));
  setType(detail::CGType::CodeplayHostTask);
}

void handler::addAccessorReq(detail::AccessorImplPtr Accessor) {
  // Add accessor to the list of requirements.
  impl->CGData.MRequirements.push_back(Accessor.get());
  // Store copy of the accessor.
  impl->CGData.MAccStorage.push_back(std::move(Accessor));
}

void handler::addLifetimeSharedPtrStorage(std::shared_ptr<const void> SPtr) {
  impl->CGData.MSharedPtrStorage.push_back(std::move(SPtr));
}

void handler::addArg(detail::kernel_param_kind_t ArgKind, void *Req,
                     int AccessTarget, int ArgIndex) {
  impl->MArgs.emplace_back(ArgKind, Req, AccessTarget, ArgIndex);
}

void handler::clearArgs() { impl->MArgs.clear(); }

void handler::setArgsToAssociatedAccessors() {
  impl->MArgs = impl->MAssociatedAccesors;
}

bool handler::HasAssociatedAccessor(detail::AccessorImplHost *Req,
                                    access::target AccessTarget) const {
  return std::find_if(
             impl->MAssociatedAccesors.cbegin(),
             impl->MAssociatedAccesors.cend(), [&](const detail::ArgDesc &AD) {
               return AD.MType == detail::kernel_param_kind_t::kind_accessor &&
                      AD.MPtr == Req &&
                      AD.MSize == static_cast<int>(AccessTarget);
             }) == impl->MAssociatedAccesors.end();
}

void handler::setType(sycl::detail::CGType Type) { impl->MCGType = Type; }
sycl::detail::CGType handler::getType() const { return impl->MCGType; }

void handler::setNDRangeDescriptorPadded(sycl::range<3> N,
                                         bool SetNumWorkGroups, int Dims) {
  impl->MNDRDesc = NDRDescT{N, SetNumWorkGroups, Dims};
}
void handler::setNDRangeDescriptorPadded(sycl::range<3> NumWorkItems,
                                         sycl::id<3> Offset, int Dims) {
  impl->MNDRDesc = NDRDescT{NumWorkItems, Offset, Dims};
}
void handler::setNDRangeDescriptorPadded(sycl::range<3> NumWorkItems,
                                         sycl::range<3> LocalSize,
                                         sycl::id<3> Offset, int Dims) {
  impl->MNDRDesc = NDRDescT{NumWorkItems, LocalSize, Offset, Dims};
}

} // namespace _V1
} // namespace sycl
