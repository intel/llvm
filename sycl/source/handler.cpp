//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stream.hpp>
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/handler_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

handler::handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
    : handler(Queue, Queue, nullptr, IsHost) {}

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 std::shared_ptr<detail::queue_impl> PrimaryQueue,
                 std::shared_ptr<detail::queue_impl> SecondaryQueue,
                 bool IsHost)
    : MQueue(std::move(Queue)), MIsHost(IsHost) {
  // Create extended members and insert handler_impl
  // TODO: When allowed to break ABI the handler_impl should be made a member
  //       of the handler class.
  auto ExtendedMembers =
      std::make_shared<std::vector<detail::ExtendedMemberT>>();
  detail::ExtendedMemberT HandlerImplMember = {
      detail::ExtendedMembersType::HANDLER_IMPL,
      std::make_shared<detail::handler_impl>(std::move(PrimaryQueue),
                                             std::move(SecondaryQueue))};
  ExtendedMembers->push_back(std::move(HandlerImplMember));
  MSharedPtrStorage.push_back(std::move(ExtendedMembers));
}

/// Gets the handler_impl at the start of the extended members.
std::shared_ptr<detail::handler_impl> handler::getHandlerImpl() const {
  std::lock_guard<std::mutex> Lock(
      detail::GlobalHandler::instance().getHandlerExtendedMembersMutex());

  assert(!MSharedPtrStorage.empty());

  std::shared_ptr<std::vector<detail::ExtendedMemberT>> ExtendedMembersVec =
      detail::convertToExtendedMembers(MSharedPtrStorage[0]);

  assert(ExtendedMembersVec->size() > 0);

  auto HandlerImplMember = (*ExtendedMembersVec)[0];

  assert(detail::ExtendedMembersType::HANDLER_IMPL == HandlerImplMember.MType);

  return std::static_pointer_cast<detail::handler_impl>(
      HandlerImplMember.MData);
}

// Sets the submission state to indicate that an explicit kernel bundle has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that a specialization constant has been set.
void handler::setStateExplicitKernelBundle() {
  getHandlerImpl()->setStateExplicitKernelBundle();
}

// Sets the submission state to indicate that a specialization constant has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that an explicit kernel bundle has been set.
void handler::setStateSpecConstSet() {
  getHandlerImpl()->setStateSpecConstSet();
}

// Returns true if the submission state is EXPLICIT_KERNEL_BUNDLE_STATE and
// false otherwise.
bool handler::isStateExplicitKernelBundle() const {
  return getHandlerImpl()->isStateExplicitKernelBundle();
}

// Returns a shared_ptr to kernel_bundle stored in the extended members vector.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns shared_ptr(nullptr) if Insert is false
std::shared_ptr<detail::kernel_bundle_impl>
handler::getOrInsertHandlerKernelBundle(bool Insert) const {

  std::lock_guard<std::mutex> Lock(
      detail::GlobalHandler::instance().getHandlerExtendedMembersMutex());

  assert(!MSharedPtrStorage.empty());

  std::shared_ptr<std::vector<detail::ExtendedMemberT>> ExtendedMembersVec =
      detail::convertToExtendedMembers(MSharedPtrStorage[0]);
  // Look for the kernel bundle in extended members
  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr;
  for (const detail::ExtendedMemberT &EMember : *ExtendedMembersVec)
    if (detail::ExtendedMembersType::HANDLER_KERNEL_BUNDLE == EMember.MType) {
      KernelBundleImpPtr =
          std::static_pointer_cast<detail::kernel_bundle_impl>(EMember.MData);
      break;
    }

  // No kernel bundle yet, create one
  if (!KernelBundleImpPtr && Insert) {
    // Create an empty kernel bundle to add kernels to later
    KernelBundleImpPtr =
        detail::getSyclObjImpl(get_kernel_bundle<bundle_state::input>(
            MQueue->get_context(), {MQueue->get_device()}, {}));

    detail::ExtendedMemberT EMember = {
        detail::ExtendedMembersType::HANDLER_KERNEL_BUNDLE, KernelBundleImpPtr};
    ExtendedMembersVec->push_back(EMember);
  }

  return KernelBundleImpPtr;
}

// Sets kernel bundle to the provided one. Either replaces existing one or
// create a new entry in the extended members vector.
void handler::setHandlerKernelBundle(
    const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr) {
  assert(!MSharedPtrStorage.empty());

  std::lock_guard<std::mutex> Lock(
      detail::GlobalHandler::instance().getHandlerExtendedMembersMutex());

  std::shared_ptr<std::vector<detail::ExtendedMemberT>> ExendedMembersVec =
      detail::convertToExtendedMembers(MSharedPtrStorage[0]);

  // Look for kernel bundle in extended members and overwrite it.
  for (detail::ExtendedMemberT &EMember : *ExendedMembersVec) {
    if (detail::ExtendedMembersType::HANDLER_KERNEL_BUNDLE == EMember.MType) {
      EMember.MData = NewKernelBundleImpPtr;
      return;
    }
  }

  // Kernel bundle was set found so we add it.
  detail::ExtendedMemberT EMember = {
      detail::ExtendedMembersType::HANDLER_KERNEL_BUNDLE,
      NewKernelBundleImpPtr};
  ExendedMembersVec->push_back(EMember);
}

event handler::finalize() {
  // This block of code is needed only for reduction implementation.
  // It is harmless (does nothing) for everything else.
  if (MIsFinalized)
    return MLastEvent;
  MIsFinalized = true;

  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr = nullptr;
  // Kernel_bundles could not be used before CGType version 1
  if (getCGTypeVersion(MCGType) >
      static_cast<unsigned int>(detail::CG::CG_VERSION::V0)) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    KernelBundleImpPtr = getOrInsertHandlerKernelBundle(/*Insert=*/false);
    if (KernelBundleImpPtr) {
      // Make sure implicit non-interop kernel bundles have the kernel
      if (!KernelBundleImpPtr->isInterop() &&
          !getHandlerImpl()->isStateExplicitKernelBundle()) {
        kernel_id KernelID =
            detail::ProgramManager::getInstance().getSYCLKernelID(MKernelName);
        bool KernelInserted =
            KernelBundleImpPtr->add_kernel(KernelID, MQueue->get_device());
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
          KernelInserted =
              KernelBundleImpPtr->add_kernel(KernelID, MQueue->get_device());
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
        assert(0 && "Expected that the bundle is either in input or executable "
                    "states.");
        break;
      }
    }
  }

  const auto &type = getType();

  if (type == detail::CG::Kernel &&
      MRequirements.size() + MEvents.size() + MStreamStorage.size() == 0) {
    // if user does not add a new dependency to the dependency graph, i.e.
    // the graph is not changed, then this faster path is used to submit kernel
    // bypassing scheduler and avoiding CommandGroup, Command objects creation.

    std::vector<RT::PiEvent> RawEvents;
    detail::EventImplPtr NewEvent;
    RT::PiEvent *OutEvent = nullptr;

    auto EnqueueKernel = [&]() {
      // 'Result' for single point of return
      cl_int Result = CL_INVALID_VALUE;

      if (MQueue->is_host()) {
        MHostKernel->call(
            MNDRDesc, (NewEvent) ? NewEvent->getHostProfilingInfo() : nullptr);
        Result = CL_SUCCESS;
      } else {
        if (MQueue->getPlugin().getBackend() ==
            backend::ext_intel_esimd_emulator) {
          MQueue->getPlugin().call<detail::PiApiKind::piEnqueueKernelLaunch>(
              nullptr, reinterpret_cast<pi_kernel>(MHostKernel.get()),
              MIsSingleTask ? 0 : MNDRDesc.Dims, &MNDRDesc.GlobalOffset[0],
              &MNDRDesc.GlobalSize[0], &MNDRDesc.LocalSize[0], 0, nullptr,
              nullptr);
          Result = CL_SUCCESS;
        } else {
          Result = enqueueImpKernel(MQueue, MNDRDesc, MArgs, KernelBundleImpPtr,
                                    MKernel, MKernelName, MOSModuleHandle,
                                    RawEvents, OutEvent, nullptr);
        }
      }
      // assert(Result != CL_INVALID_VALUE);
      return Result;
    };

    bool DiscardEvent = false;
    if (MQueue->has_discard_events_support()) {
      // Kernel only uses assert if it's non interop one
      bool KernelUsesAssert =
          !(MKernel && MKernel->isInterop()) &&
          detail::ProgramManager::getInstance().kernelUsesAssert(
              MOSModuleHandle, MKernelName);
      DiscardEvent = !KernelUsesAssert;
    }

    if (DiscardEvent) {
      if (CL_SUCCESS != EnqueueKernel())
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
    } else {
      NewEvent = std::make_shared<detail::event_impl>(MQueue);
      NewEvent->setContextImpl(MQueue->getContextImplPtr());
      OutEvent = &NewEvent->getHandleRef();

      if (CL_SUCCESS != EnqueueKernel())
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
      else if (NewEvent->is_host() || NewEvent->getHandleRef() == nullptr)
        NewEvent->setComplete();

      MLastEvent = detail::createSyclObjFromImpl<event>(NewEvent);
    }
    return MLastEvent;
  }

  std::unique_ptr<detail::CG> CommandGroup;
  switch (type) {
  case detail::CG::Kernel:
  case detail::CG::RunOnHostIntel: {
    // Copy kernel name here instead of move so that it's available after
    // running of this method by reductions implementation. This allows for
    // assert feature to check if kernel uses assertions
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(MIsSingleTask), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), std::move(MArgs),
        MKernelName, MOSModuleHandle, std::move(MStreamStorage), MCGType,
        MCodeLoc));
    break;
  }
  case detail::CG::CodeplayInteropTask:
    CommandGroup.reset(new detail::CGInteropTask(
        std::move(MInteropTask), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::CopyAccToPtr:
  case detail::CG::CopyPtrToAcc:
  case detail::CG::CopyAccToAcc:
    CommandGroup.reset(new detail::CGCopy(
        MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::Fill:
    CommandGroup.reset(new detail::CGFill(
        std::move(MPattern), MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::UpdateHost:
    CommandGroup.reset(new detail::CGUpdateHost(
        MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::CopyUSM:
    CommandGroup.reset(new detail::CGCopyUSM(
        MSrcPtr, MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::FillUSM:
    CommandGroup.reset(new detail::CGFillUSM(
        std::move(MPattern), MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::PrefetchUSM:
    CommandGroup.reset(new detail::CGPrefetchUSM(
        MDstPtr, MLength, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::AdviseUSM:
    CommandGroup.reset(new detail::CGAdviseUSM(
        MDstPtr, MLength, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::CodeplayHostTask:
    CommandGroup.reset(new detail::CGHostTask(
        std::move(MHostTask), MQueue, MQueue->getContextImplPtr(),
        std::move(MArgs), std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::Barrier:
  case detail::CG::BarrierWaitlist:
    CommandGroup.reset(new detail::CGBarrier(
        std::move(MEventsWaitWithBarrier), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::None:
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      std::cout << "WARNING: An empty command group is submitted." << std::endl;
    }
    detail::EventImplPtr Event =
        std::make_shared<cl::sycl::detail::event_impl>();
    MLastEvent = detail::createSyclObjFromImpl<event>(Event);
    return MLastEvent;
  }

  if (!CommandGroup)
    throw sycl::runtime_error(
        "Internal Error. Command group cannot be constructed.",
        PI_INVALID_OPERATION);

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);
  return MLastEvent;
}

void handler::associateWithHandler(detail::AccessorBaseHost *AccBase,
                                   access::target AccTarget) {
  detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
  detail::Requirement *Req = AccImpl.get();
  // Add accessor to the list of requirements.
  MRequirements.push_back(Req);
  // Store copy of the accessor.
  MAccStorage.push_back(std::move(AccImpl));
  // Add an accessor to the handler list of associated accessors.
  // For associated accessors index does not means nothing.
  MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                   Req, static_cast<int>(AccTarget),
                                   /*index*/ 0);
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

// TODO remove this one once ABI breaking changes are allowed.
void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource) {
  processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource,
             false);
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
    const access::target AccTarget = static_cast<access::target>(Size & 0x7ff);
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
      MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                         SizeInBytes, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
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
    case access::target::host_buffer: {
      throw cl::sycl::invalid_parameter_error(
          "Unsupported accessor target case.", PI_INVALID_OPERATION);
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
    throw runtime_error("Invalid kernel param kind", PI_INVALID_VALUE);
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

// TODO remove once ABI breaking changes are allowed
void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs) {
  extractArgsAndReqsFromLambda(LambdaPtr, KernelArgsNum, KernelArgs, false);
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
          static_cast<access::target>(Size & 0x7ff);
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
std::string handler::getKernelName() {
  return MKernel->get_info<info::kernel::function_name>();
}

void handler::verifyUsedKernelBundle(const std::string &KernelName) {
  auto UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundle(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  if (!getHandlerImpl()->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  device Dev = detail::getDeviceFromHandler(*this);
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

__SYCL2020_DEPRECATED("use 'ext_oneapi_barrier' instead")
void handler::barrier(const std::vector<event> &WaitList) {
  handler::ext_oneapi_barrier(WaitList);
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
  setType(detail::CG::AdviseUSM);

  assert(!MSharedPtrStorage.empty());

  std::lock_guard<std::mutex> Lock(
      detail::GlobalHandler::instance().getHandlerExtendedMembersMutex());

  std::shared_ptr<std::vector<detail::ExtendedMemberT>> ExtendedMembersVec =
      detail::convertToExtendedMembers(MSharedPtrStorage[0]);

  detail::ExtendedMemberT EMember = {
      detail::ExtendedMembersType::HANDLER_MEM_ADVICE,
      std::make_shared<pi_mem_advice>(pi_mem_advice(Advice))};

  ExtendedMembersVec->push_back(EMember);
}

void handler::use_kernel_bundle(
    const kernel_bundle<bundle_state::executable> &ExecBundle) {

  std::shared_ptr<detail::queue_impl> PrimaryQueue =
      getHandlerImpl()->MSubmissionPrimaryQueue;
  if (PrimaryQueue->get_context() != ExecBundle.get_context())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the primary queue is different from the "
        "context associated with the kernel bundle");

  std::shared_ptr<detail::queue_impl> SecondaryQueue =
      getHandlerImpl()->MSubmissionSecondaryQueue;
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
  MEvents.push_back(EventImpl);
}

void handler::depends_on(const std::vector<event> &Events) {
  for (const event &Event : Events) {
    auto EventImpl = detail::getSyclObjImpl(Event);
    if (EventImpl->isDiscarded()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Queue operation cannot depend on discarded event.");
    }
    MEvents.push_back(EventImpl);
  }
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
