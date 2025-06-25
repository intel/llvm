//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/detail/helpers.hpp"
#include "ur_api.h"
#include <algorithm>

#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/graph/dynamic_impl.hpp>
#include <detail/graph/graph_impl.hpp>
#include <detail/graph/node_impl.hpp>
#include <detail/handler_impl.hpp>
#include <detail/helpers.hpp>
#include <detail/host_task.hpp>
#include <detail/image_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/ur_info_code.hpp>
#include <detail/usm/usm_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stream.hpp>

#include <sycl/ext/oneapi/bindless_images_memory.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
// TODO: Check if two ABI exports below are still necessary.
#endif
device_impl &getDeviceImplFromHandler(handler &CGH) {
  return getSyclObjImpl(CGH)->get_device();
}

device getDeviceFromHandler(handler &CGH) {
  return createSyclObjFromImpl<device>(getSyclObjImpl(CGH)->get_device());
}

bool isDeviceGlobalUsedInKernel(const void *DeviceGlobalPtr) {
  DeviceGlobalMapEntry *DGEntry =
      detail::ProgramManager::getInstance().getDeviceGlobalEntry(
          DeviceGlobalPtr);
  return DGEntry && !DGEntry->MImageIdentifiers.empty();
}

static ur_exp_image_copy_flags_t
getUrImageCopyFlags(sycl::usm::alloc SrcPtrType, sycl::usm::alloc DstPtrType) {
  if (DstPtrType == sycl::usm::alloc::device) {
    // Dest is on device
    if (SrcPtrType == sycl::usm::alloc::device)
      return UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE;
    if (SrcPtrType == sycl::usm::alloc::host ||
        SrcPtrType == sycl::usm::alloc::unknown)
      return UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE;
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unknown copy source location");
  }
  if (DstPtrType == sycl::usm::alloc::host ||
      DstPtrType == sycl::usm::alloc::unknown) {
    // Dest is on host
    if (SrcPtrType == sycl::usm::alloc::device)
      return UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST;
    if (SrcPtrType == sycl::usm::alloc::host ||
        SrcPtrType == sycl::usm::alloc::unknown)
      return UR_EXP_IMAGE_COPY_FLAG_HOST_TO_HOST;
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

// Bindless image helpers

// Fill image type and return depth or array_size
static unsigned int
fill_image_type(const ext::oneapi::experimental::image_descriptor &Desc,
                ur_image_desc_t &UrDesc) {
  if (Desc.array_size > 1) {
    // Image Array.
    UrDesc.type =
        Desc.height > 0 ? UR_MEM_TYPE_IMAGE2D_ARRAY : UR_MEM_TYPE_IMAGE1D_ARRAY;

    // Cubemap.
    UrDesc.type =
        Desc.type == sycl::ext::oneapi::experimental::image_type::cubemap
            ? UR_MEM_TYPE_IMAGE_CUBEMAP_EXP
        : Desc.type == sycl::ext::oneapi::experimental::image_type::gather
            ? UR_MEM_TYPE_IMAGE_GATHER_EXP
            : UrDesc.type;

    return Desc.array_size;
  }

  UrDesc.type = Desc.depth > 0 ? UR_MEM_TYPE_IMAGE3D
                               : (Desc.height > 0 ? UR_MEM_TYPE_IMAGE2D
                                                  : UR_MEM_TYPE_IMAGE1D);
  return Desc.depth;
}

// Fill image format
static ur_image_format_t
fill_format(const ext::oneapi::experimental::image_descriptor &Desc) {
  ur_image_format_t PiFormat;

  PiFormat.channelType =
      sycl::_V1::detail::convertChannelType(Desc.channel_type);
  PiFormat.channelOrder = sycl::detail::convertChannelOrder(
      sycl::_V1::ext::oneapi::experimental::detail::
          get_image_default_channel_order(Desc.num_channels));

  return PiFormat;
}

static void
verify_copy(const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
            const ext::oneapi::experimental::image_descriptor &DestImgDesc) {

  if (SrcImgDesc.width != DestImgDesc.width ||
      SrcImgDesc.height != DestImgDesc.height ||
      SrcImgDesc.depth != DestImgDesc.depth) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: The source image and the destination "
                          "image must have equal dimensions!");
  }

  if (SrcImgDesc.num_channels != DestImgDesc.num_channels) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: The source image and the destination "
                          "image must have the same number of channels!");
  }
}

static void
verify_sub_copy(const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
                sycl::range<3> SrcOffset,
                const ext::oneapi::experimental::image_descriptor &DestImgDesc,
                sycl::range<3> DestOffset, sycl::range<3> CopyExtent) {

  auto isOutOfRange = [](const sycl::range<3> &range,
                         const sycl::range<3> &offset,
                         const sycl::range<3> &copyExtent) {
    sycl::range<3> result = (range > 0UL && ((offset + copyExtent) > range));

    return (static_cast<bool>(result[0]) || static_cast<bool>(result[1]) ||
            static_cast<bool>(result[2]));
  };

  sycl::range<3> SrcImageSize = {SrcImgDesc.width, SrcImgDesc.height,
                                 SrcImgDesc.depth};
  sycl::range<3> DestImageSize = {DestImgDesc.width, DestImgDesc.height,
                                  DestImgDesc.depth};

  if (isOutOfRange(SrcImageSize, SrcOffset, CopyExtent) ||
      isOutOfRange(DestImageSize, DestOffset, CopyExtent)) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Copy Error: Image copy attempted to access out of bounds memory!");
  }

  if (SrcImgDesc.num_channels != DestImgDesc.num_channels) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: The source image and the destination "
                          "image must have the same number of channels!");
  }
}

static ur_image_desc_t
fill_image_desc(const ext::oneapi::experimental::image_descriptor &ImgDesc) {
  ur_image_desc_t UrDesc = {};
  UrDesc.stype = UR_STRUCTURE_TYPE_IMAGE_DESC;
  UrDesc.width = ImgDesc.width;
  UrDesc.height = ImgDesc.height;
  UrDesc.depth = ImgDesc.depth;
  UrDesc.arraySize = ImgDesc.array_size;
  return UrDesc;
}

static void
fill_copy_args(detail::handler_impl *impl,
               const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
               const ext::oneapi::experimental::image_descriptor &DestImgDesc,
               ur_exp_image_copy_flags_t ImageCopyFlags, size_t SrcPitch,
               size_t DestPitch, sycl::range<3> SrcOffset = {0, 0, 0},
               sycl::range<3> SrcExtent = {0, 0, 0},
               sycl::range<3> DestOffset = {0, 0, 0},
               sycl::range<3> DestExtent = {0, 0, 0},
               sycl::range<3> CopyExtent = {0, 0, 0}) {
  SrcImgDesc.verify();
  DestImgDesc.verify();

  // CopyExtent.size() should only be greater than 0 when sub-copy is occurring.
  if (CopyExtent.size() == 0) {
    detail::verify_copy(SrcImgDesc, DestImgDesc);
  } else {
    detail::verify_sub_copy(SrcImgDesc, SrcOffset, DestImgDesc, DestOffset,
                            CopyExtent);
  }

  ur_image_desc_t UrSrcDesc = detail::fill_image_desc(SrcImgDesc);
  ur_image_desc_t UrDestDesc = detail::fill_image_desc(DestImgDesc);
  ur_image_format_t UrSrcFormat = detail::fill_format(SrcImgDesc);
  ur_image_format_t UrDestFormat = detail::fill_format(DestImgDesc);
  auto ZCopyExtentComponent = detail::fill_image_type(SrcImgDesc, UrSrcDesc);
  detail::fill_image_type(DestImgDesc, UrDestDesc);

  impl->MSrcOffset = {SrcOffset[0], SrcOffset[1], SrcOffset[2]};
  impl->MDestOffset = {DestOffset[0], DestOffset[1], DestOffset[2]};
  impl->MSrcImageDesc = UrSrcDesc;
  impl->MDstImageDesc = UrDestDesc;
  impl->MSrcImageFormat = UrSrcFormat;
  impl->MDstImageFormat = UrDestFormat;
  impl->MImageCopyFlags = ImageCopyFlags;

  if (CopyExtent.size() != 0) {
    impl->MCopyExtent = {CopyExtent[0], CopyExtent[1], CopyExtent[2]};
  } else {
    impl->MCopyExtent = {SrcImgDesc.width, SrcImgDesc.height,
                         ZCopyExtentComponent};
  }

  if (SrcExtent.size() != 0) {
    impl->MSrcImageDesc.width = SrcExtent[0];
    impl->MSrcImageDesc.height = SrcExtent[1];
    impl->MSrcImageDesc.depth = SrcExtent[2];
  }

  if (DestExtent.size() != 0) {
    impl->MDstImageDesc.width = DestExtent[0];
    impl->MDstImageDesc.height = DestExtent[1];
    impl->MDstImageDesc.depth = DestExtent[2];
  }

  if (impl->MImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
    impl->MSrcImageDesc.rowPitch = 0;
    impl->MDstImageDesc.rowPitch = DestPitch;
  } else if (impl->MImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
    impl->MSrcImageDesc.rowPitch = SrcPitch;
    impl->MDstImageDesc.rowPitch = 0;
  } else {
    impl->MSrcImageDesc.rowPitch = SrcPitch;
    impl->MDstImageDesc.rowPitch = DestPitch;
  }
}

static void
fill_copy_args(detail::handler_impl *impl,
               const ext::oneapi::experimental::image_descriptor &Desc,
               ur_exp_image_copy_flags_t ImageCopyFlags,
               sycl::range<3> SrcOffset = {0, 0, 0},
               sycl::range<3> SrcExtent = {0, 0, 0},
               sycl::range<3> DestOffset = {0, 0, 0},
               sycl::range<3> DestExtent = {0, 0, 0},
               sycl::range<3> CopyExtent = {0, 0, 0}) {

  fill_copy_args(impl, Desc, Desc, ImageCopyFlags, 0 /*SrcPitch*/,
                 0 /*DestPitch*/, SrcOffset, SrcExtent, DestOffset, DestExtent,
                 CopyExtent);
}

static void
fill_copy_args(detail::handler_impl *impl,
               const ext::oneapi::experimental::image_descriptor &Desc,
               ur_exp_image_copy_flags_t ImageCopyFlags, size_t SrcPitch,
               size_t DestPitch, sycl::range<3> SrcOffset = {0, 0, 0},
               sycl::range<3> SrcExtent = {0, 0, 0},
               sycl::range<3> DestOffset = {0, 0, 0},
               sycl::range<3> DestExtent = {0, 0, 0},
               sycl::range<3> CopyExtent = {0, 0, 0}) {

  fill_copy_args(impl, Desc, Desc, ImageCopyFlags, SrcPitch, DestPitch,
                 SrcOffset, SrcExtent, DestOffset, DestExtent, CopyExtent);
}

static void
fill_copy_args(detail::handler_impl *impl,
               const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
               const ext::oneapi::experimental::image_descriptor &DestImgDesc,
               ur_exp_image_copy_flags_t ImageCopyFlags,
               sycl::range<3> SrcOffset = {0, 0, 0},
               sycl::range<3> SrcExtent = {0, 0, 0},
               sycl::range<3> DestOffset = {0, 0, 0},
               sycl::range<3> DestExtent = {0, 0, 0},
               sycl::range<3> CopyExtent = {0, 0, 0}) {

  fill_copy_args(impl, SrcImgDesc, DestImgDesc, ImageCopyFlags, 0 /*SrcPitch*/,
                 0 /*DestPitch*/, SrcOffset, SrcExtent, DestOffset, DestExtent,
                 CopyExtent);
}

} // namespace detail

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
handler::handler(detail::handler_impl &HandlerImpl) : impl(&HandlerImpl) {}
#else
handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 bool CallerNeedsEvent)
    : impl(std::make_shared<detail::handler_impl>(*Queue, nullptr,
                                                  CallerNeedsEvent)),
      MQueueDoNotUse(std::move(Queue)) {}

handler::handler(
    std::shared_ptr<detail::queue_impl> Queue,
    [[maybe_unused]] std::shared_ptr<detail::queue_impl> PrimaryQueue,
    std::shared_ptr<detail::queue_impl> SecondaryQueue, bool CallerNeedsEvent)
    : impl(std::make_shared<detail::handler_impl>(*Queue, SecondaryQueue.get(),
                                                  CallerNeedsEvent)),
      MQueueDoNotUse(Queue) {}

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 detail::queue_impl *SecondaryQueue, bool CallerNeedsEvent)
    : impl(std::make_shared<detail::handler_impl>(*Queue, SecondaryQueue,
                                                  CallerNeedsEvent)),
      MQueueDoNotUse(std::move(Queue)) {}

handler::handler(
    std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph)
    : impl(std::make_shared<detail::handler_impl>(*Graph)) {}

#endif

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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// Returns a shared_ptr to the kernel_bundle.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns shared_ptr(nullptr) if Insert is false
std::shared_ptr<detail::kernel_bundle_impl>
handler::getOrInsertHandlerKernelBundle(bool Insert) const {
  if (impl->MKernelBundle || !Insert)
    return impl->MKernelBundle;

  context Ctx = detail::createSyclObjFromImpl<context>(impl->get_context());
  impl->MKernelBundle =
      detail::getSyclObjImpl(get_kernel_bundle<bundle_state::input>(
          Ctx, {detail::createSyclObjFromImpl<device>(impl->get_device())},
          {}));
  return impl->MKernelBundle;
}
#endif

// Returns a ptr to the kernel_bundle.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns nullptr if Insert is false
detail::kernel_bundle_impl *
handler::getOrInsertHandlerKernelBundlePtr(bool Insert) const {
  if (impl->MKernelBundle || !Insert)
    return impl->MKernelBundle.get();

  context Ctx = detail::createSyclObjFromImpl<context>(impl->get_context());
  impl->MKernelBundle =
      detail::getSyclObjImpl(get_kernel_bundle<bundle_state::input>(
          Ctx, {detail::createSyclObjFromImpl<device>(impl->get_device())},
          {}));
  return impl->MKernelBundle.get();
}

// Sets kernel bundle to the provided one.
template <typename SharedPtrT>
void handler::setHandlerKernelBundle(SharedPtrT &&NewKernelBundleImpPtr) {
  impl->MKernelBundle = std::forward<SharedPtrT>(NewKernelBundleImpPtr);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void handler::setHandlerKernelBundle(
    const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr) {
  impl->MKernelBundle = NewKernelBundleImpPtr;
}
#endif

void handler::setHandlerKernelBundle(kernel Kernel) {
  // Kernel may not have an associated kernel bundle if it is created from a
  // program. As such, apply getSyclObjImpl directly on the kernel, i.e. not
  //  the other way around: getSyclObjImp(Kernel->get_kernel_bundle()).
  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpl =
      detail::getSyclObjImpl(Kernel)->get_kernel_bundle();
  setHandlerKernelBundle(std::move(KernelBundleImpl));
}

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
detail::EventImplPtr handler::finalize() {
#else
event handler::finalize() {
#endif
  // This block of code is needed only for reduction implementation.
  // It is harmless (does nothing) for everything else.
  if (MIsFinalized)
    return MLastEvent;
  MIsFinalized = true;

  const auto &type = getType();
  detail::queue_impl *Queue = impl->get_queue_or_null();
  ext::oneapi::experimental::detail::graph_impl *Graph =
      impl->get_graph_or_null();
  const bool KernelFastPath =
      (Queue && !Graph && !impl->MSubgraphNode && !Queue->hasCommandGraph() &&
       !impl->CGData.MRequirements.size() && !MStreamStorage.size() &&
       detail::Scheduler::areEventsSafeForSchedulerBypass(
           impl->CGData.MEvents, Queue->getContextImpl()));

  // Extract arguments from the kernel lambda, if required.
  // Skipping this is currently limited to simple kernels on the fast path.
  if (type == detail::CGType::Kernel && impl->MKernelFuncPtr &&
      (!KernelFastPath || impl->MKernelHasSpecialCaptures)) {
    clearArgs();
    extractArgsAndReqsFromLambda((char *)impl->MKernelFuncPtr,
                                 impl->MKernelParamDescGetter,
                                 impl->MKernelNumArgs, impl->MKernelIsESIMD);
  }

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
          if ((Acc.MType == detail::kernel_param_kind_t::kind_accessor) &&
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

  if (type == detail::CGType::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    detail::kernel_bundle_impl *KernelBundleImpPtr =
        getOrInsertHandlerKernelBundlePtr(/*Insert=*/false);
    if (KernelBundleImpPtr) {
      // Make sure implicit non-interop kernel bundles have the kernel
      if (!impl->isStateExplicitKernelBundle() &&
          !(MKernel && MKernel->isInterop()) &&
          (KernelBundleImpPtr->empty() ||
           KernelBundleImpPtr->hasSYCLOfflineImages()) &&
          !KernelBundleImpPtr->tryGetKernel(toKernelNameStrT(MKernelName))) {
        detail::device_impl &Dev = impl->get_device();
        kernel_id KernelID =
            detail::ProgramManager::getInstance().getSYCLKernelID(
                toKernelNameStrT(MKernelName));
        bool KernelInserted = KernelBundleImpPtr->add_kernel(
            KernelID, detail::createSyclObjFromImpl<device>(Dev));
        // If kernel was not inserted and the bundle is in input mode we try
        // building it and trying to find the kernel in executable mode
        if (!KernelInserted &&
            KernelBundleImpPtr->get_bundle_state() == bundle_state::input) {
          auto KernelBundle =
              detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
                  *KernelBundleImpPtr);
          kernel_bundle<bundle_state::executable> ExecKernelBundle =
              build(KernelBundle);
          KernelBundleImpPtr = detail::getSyclObjImpl(ExecKernelBundle).get();
          // Raw ptr KernelBundleImpPtr is valid, because we saved the
          // shared_ptr to the handler
          setHandlerKernelBundle(KernelBundleImpPtr->shared_from_this());
          KernelInserted = KernelBundleImpPtr->add_kernel(
              KernelID, detail::createSyclObjFromImpl<device>(Dev));
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
                *KernelBundleImpPtr));
        KernelBundleImpPtr = detail::getSyclObjImpl(ExecBundle).get();
        // Raw ptr KernelBundleImpPtr is valid, because we saved the shared_ptr
        // to the handler
        setHandlerKernelBundle(KernelBundleImpPtr->shared_from_this());
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

    if (KernelFastPath) {
      // if user does not add a new dependency to the dependency graph, i.e.
      // the graph is not changed, then this faster path is used to submit
      // kernel bypassing scheduler and avoiding CommandGroup, Command objects
      // creation.
      std::vector<ur_event_handle_t> RawEvents = detail::Command::getUrEvents(
          impl->CGData.MEvents, impl->get_queue_or_null(), false);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
      detail::EventImplPtr &LastEventImpl = MLastEvent;
#else
      const detail::EventImplPtr &LastEventImpl =
          detail::getSyclObjImpl(MLastEvent);
#endif

      bool DiscardEvent =
          !impl->MEventNeeded && impl->get_queue().supportsDiscardingPiEvents();
      if (DiscardEvent) {
        // Kernel only uses assert if it's non interop one
        bool KernelUsesAssert =
            !(MKernel && MKernel->isInterop()) &&
            detail::ProgramManager::getInstance().kernelUsesAssert(
                toKernelNameStrT(MKernelName), impl->MKernelNameBasedCachePtr);
        DiscardEvent = !KernelUsesAssert;
      }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
      if (!DiscardEvent) {
        LastEventImpl = detail::event_impl::create_completed_host_event();
      }
#endif

#ifdef XPTI_ENABLE_INSTRUMENTATION
      const bool xptiEnabled = xptiTraceEnabled();
#endif
      auto EnqueueKernel = [&]() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
        int32_t StreamID = xpti::invalid_id<>;
        xpti_td *CmdTraceEvent = nullptr;
        uint64_t InstanceID = 0;
        if (xptiEnabled) {
          StreamID = xptiRegisterStream(detail::SYCL_STREAM_NAME);
          std::tie(CmdTraceEvent, InstanceID) = emitKernelInstrumentationData(
              StreamID, MKernel, MCodeLoc, impl->MIsTopCodeLoc,
              MKernelName.data(), impl->MKernelNameBasedCachePtr,
              impl->get_queue_or_null(), impl->MNDRDesc, KernelBundleImpPtr,
              impl->MArgs);
          detail::emitInstrumentationGeneral(StreamID, InstanceID,
                                             CmdTraceEvent,
                                             xpti::trace_task_begin, nullptr);
        }
#endif
        const detail::RTDeviceBinaryImage *BinImage = nullptr;
        if (detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_KERNELS>::get()) {
          std::tie(BinImage, std::ignore) = detail::retrieveKernelBinary(
              impl->get_queue(), toKernelNameStrT(MKernelName));
          assert(BinImage && "Failed to obtain a binary image.");
        }
        enqueueImpKernel(
            impl->get_queue(), impl->MNDRDesc, impl->MArgs, KernelBundleImpPtr,
            MKernel.get(), toKernelNameStrT(MKernelName),
            impl->MKernelNameBasedCachePtr, RawEvents,
            DiscardEvent ? nullptr : LastEventImpl.get(), nullptr,
            impl->MKernelCacheConfig, impl->MKernelIsCooperative,
            impl->MKernelUsesClusterLaunch, impl->MKernelWorkGroupMemorySize,
            BinImage, impl->MKernelFuncPtr, impl->MKernelNumArgs,
            impl->MKernelParamDescGetter, impl->MKernelHasSpecialCaptures);
#ifdef XPTI_ENABLE_INSTRUMENTATION
        if (xptiEnabled) {
          // Emit signal only when event is created
          if (!DiscardEvent) {
            detail::emitInstrumentationGeneral(
                StreamID, InstanceID, CmdTraceEvent, xpti::trace_signal,
                static_cast<const void *>(LastEventImpl->getHandle()));
          }
          detail::emitInstrumentationGeneral(StreamID, InstanceID,
                                             CmdTraceEvent,
                                             xpti::trace_task_end, nullptr);
        }
#endif
      };

      if (DiscardEvent) {
        EnqueueKernel();
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
        LastEventImpl->setStateDiscarded();
#endif
      } else {
        detail::queue_impl &Queue = impl->get_queue();
        LastEventImpl->setQueue(Queue);
        LastEventImpl->setWorkerQueue(Queue.weak_from_this());
        LastEventImpl->setContextImpl(impl->get_context());
        LastEventImpl->setStateIncomplete();
        LastEventImpl->setSubmissionTime();

        EnqueueKernel();
        LastEventImpl->setEnqueued();
        // connect returned event with dependent events
        if (!Queue.isInOrder()) {
          // MEvents is not used anymore, so can move.
          LastEventImpl->getPreparedDepsEvents() =
              std::move(impl->CGData.MEvents);
          // LastEventImpl is local for current thread, no need to lock.
          LastEventImpl->cleanDepEventsThroughOneLevelUnlocked();
        }
      }
      return MLastEvent;
    }
  }

  std::unique_ptr<detail::CG> CommandGroup;
  switch (type) {
  case detail::CGType::Kernel: {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    // Copy kernel name here instead of move so that it's available after
    // running of this method by reductions implementation. This allows for
    // assert feature to check if kernel uses assertions
#endif
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(impl->MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(impl->MKernelBundle), std::move(impl->CGData),
        std::move(impl->MArgs), toKernelNameStrT(MKernelName),
        impl->MKernelNameBasedCachePtr, std::move(MStreamStorage),
        std::move(impl->MAuxiliaryResources), getType(),
        impl->MKernelCacheConfig, impl->MKernelIsCooperative,
        impl->MKernelUsesClusterLaunch, impl->MKernelWorkGroupMemorySize,
        MCodeLoc));
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
    detail::context_impl &Context = impl->get_context();
    detail::queue_impl *Queue = impl->get_queue_or_null();
    CommandGroup.reset(new detail::CGHostTask(
        std::move(impl->MHostTask), Queue, Context.shared_from_this(),
        std::move(impl->MArgs), std::move(impl->CGData), getType(), MCodeLoc));
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
      CommandGroup.reset(new detail::CGBarrier(
          std::move(impl->MEventsWaitWithBarrier), impl->MEventMode,
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
    detail::queue_impl *Queue = impl->get_queue_or_null();
    std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> ParentGraph =
        Queue ? Queue->getCommandGraph() : impl->get_graph().shared_from_this();

    // If a parent graph is set that means we are adding or recording a subgraph
    // and we don't want to actually execute this command graph submission.
    if (ParentGraph) {
      ext::oneapi::experimental::detail::graph_impl::WriteLock ParentLock;
      if (Queue) {
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
      detail::queue_impl &Queue = impl->get_queue();
      bool DiscardEvent = !impl->MEventNeeded &&
                          Queue.supportsDiscardingPiEvents() &&
                          !impl->MExecGraph->containsHostTask();
      detail::EventImplPtr GraphCompletionEvent = impl->MExecGraph->enqueue(
          Queue, std::move(impl->CGData), !DiscardEvent);
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
      return GraphCompletionEvent;
#else
      return sycl::detail::createSyclObjFromImpl<sycl::event>(
          GraphCompletionEvent
              ? GraphCompletionEvent
              : sycl::detail::event_impl::create_discarded_event());
#endif
    }
  } break;
  case detail::CGType::CopyImage:
    CommandGroup.reset(new detail::CGCopyImage(
        MSrcPtr, MDstPtr, impl->MSrcImageDesc, impl->MDstImageDesc,
        impl->MSrcImageFormat, impl->MDstImageFormat, impl->MImageCopyFlags,
        impl->MSrcOffset, impl->MDestOffset, impl->MCopyExtent,
        std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::SemaphoreWait:
    CommandGroup.reset(
        new detail::CGSemaphoreWait(impl->MExternalSemaphore, impl->MWaitValue,
                                    std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::SemaphoreSignal:
    CommandGroup.reset(new detail::CGSemaphoreSignal(
        impl->MExternalSemaphore, impl->MSignalValue, std::move(impl->CGData),
        MCodeLoc));
    break;
  case detail::CGType::AsyncAlloc:
    CommandGroup.reset(new detail::CGAsyncAlloc(
        impl->MAsyncAllocEvent, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::AsyncFree:
    CommandGroup.reset(new detail::CGAsyncFree(
        impl->MFreePtr, std::move(impl->CGData), MCodeLoc));
    break;
  case detail::CGType::None:
    CommandGroup.reset(new detail::CG(detail::CGType::None,
                                      std::move(impl->CGData), MCodeLoc));
    break;
  }

  if (!CommandGroup)
    throw exception(make_error_code(errc::runtime),
                    "Internal Error. Command group cannot be constructed.");

  // Propagate MIsTopCodeLoc state to CommandGroup.
  // Will be used for XPTI payload generation for CG's related events.
  CommandGroup->MIsTopCodeLoc = impl->MIsTopCodeLoc;

  // If there is a graph associated with the handler we are in the explicit
  // graph mode, so we store the CG instead of submitting it to the scheduler,
  // so it can be retrieved by the graph later.
  if (impl->get_graph_or_null()) {
    impl->MGraphNodeCG = std::move(CommandGroup);
    auto EventImpl = detail::event_impl::create_completed_host_event();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    return EventImpl;
#else
    return detail::createSyclObjFromImpl<event>(EventImpl);
#endif
  }

  // Because graph case is handled right above.
  assert(Queue);

  // If the queue has an associated graph then we need to take the CG and pass
  // it to the graph to create a node, rather than submit it to the scheduler.
  if (auto GraphImpl = Queue->getCommandGraph(); GraphImpl) {
    auto EventImpl = detail::event_impl::create_completed_host_event();
    EventImpl->setSubmittedQueue(Queue->weak_from_this());
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
    if (Queue->isInOrder()) {
      // In-order queues create implicit linear dependencies between nodes.
      // Find the last node added to the graph from this queue, so our new
      // node can set it as a predecessor.
      std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
          Deps;
      if (auto DependentNode = GraphImpl->getLastInorderNode(Queue)) {
        Deps.push_back(std::move(DependentNode));
      }
      NodeImpl = GraphImpl->add(NodeType, std::move(CommandGroup), Deps);

      // If we are recording an in-order queue remember the new node, so it
      // can be used as a dependency for any more nodes recorded from this
      // queue.
      GraphImpl->setLastInorderNode(*Queue, NodeImpl);
    } else {
      auto LastBarrierRecordedFromQueue =
          GraphImpl->getBarrierDep(Queue->weak_from_this());
      std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
          Deps;

      if (LastBarrierRecordedFromQueue) {
        Deps.push_back(LastBarrierRecordedFromQueue);
      }
      NodeImpl = GraphImpl->add(NodeType, std::move(CommandGroup), Deps);

      if (NodeImpl->MCGType == sycl::detail::CGType::Barrier) {
        GraphImpl->setBarrierDep(Queue->weak_from_this(), NodeImpl);
      }
    }

    // Associate an event with this new node and return the event.
    GraphImpl->addEventForNode(EventImpl, std::move(NodeImpl));

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    return EventImpl;
#else
    return detail::createSyclObjFromImpl<event>(EventImpl);
#endif
  }

  bool DiscardEvent = !impl->MEventNeeded && Queue &&
                      Queue->supportsDiscardingPiEvents() &&
                      CommandGroup->getRequirements().size() == 0;

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), Queue->shared_from_this(), !DiscardEvent);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  MLastEvent = DiscardEvent ? nullptr : Event;
#else
  MLastEvent = detail::createSyclObjFromImpl<event>(Event);
#endif
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

static void addArgsForLocalAccessor(detail::LocalAccessorImplHost *LAcc,
                                    size_t Index, size_t &IndexShift,
                                    bool IsKernelCreatedFromSource,
                                    std::vector<detail::ArgDesc> &Args,
                                    bool IsESIMD) {
  using detail::kernel_param_kind_t;

  range<3> &LAccSize = LAcc->MSize;
  const int Dims = LAcc->MDims;
  int SizeInBytes = LAcc->MElemSize;
  for (int I = 0; I < Dims; ++I)
    SizeInBytes *= LAccSize[I];

  // Some backends do not accept zero-sized local memory arguments, so we
  // make it a minimum allocation of 1 byte.
  SizeInBytes = std::max(SizeInBytes, 1);
  Args.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr, SizeInBytes,
                    Index + IndexShift);
  // TODO ESIMD currently does not suport MSize field passing yet
  // accessor::init for ESIMD-mode accessor has a single field, translated
  // to a single kernel argument set above.
  if (!IsESIMD && !IsKernelCreatedFromSource) {
    ++IndexShift;
    const size_t SizeAccField = (Dims == 0 ? 1 : Dims) * sizeof(LAccSize[0]);
    Args.emplace_back(kernel_param_kind_t::kind_std_layout, &LAccSize,
                      SizeAccField, Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout, &LAccSize,
                      SizeAccField, Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout, &LAccSize,
                      SizeAccField, Index + IndexShift);
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
      detail::LocalAccessorImplHost *LAccImpl =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      addArgsForLocalAccessor(LAccImpl, Index, IndexShift,
                              IsKernelCreatedFromSource, impl->MArgs, IsESIMD);
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
  case kernel_param_kind_t::kind_dynamic_accessor: {
    const access::target AccTarget =
        static_cast<access::target>(Size & AccessTargetMask);
    switch (AccTarget) {
    case access::target::local: {

      // We need to recover the inheritance layout by casting to
      // dynamic_parameter_impl first. Casting directly to
      // dynamic_local_accessor_impl would result in an incorrect pointer.
      auto *DynParamImpl = static_cast<
          ext::oneapi::experimental::detail::dynamic_parameter_impl *>(Ptr);

      registerDynamicParameter(DynParamImpl, Index + IndexShift);

      auto *DynLocalAccessorImpl = static_cast<
          ext::oneapi::experimental::detail::dynamic_local_accessor_impl *>(
          DynParamImpl);

      addArgsForLocalAccessor(&DynLocalAccessorImpl->LAccImplHost, Index,
                              IndexShift, IsKernelCreatedFromSource,
                              impl->MArgs, IsESIMD);
      break;
    }
    default: {
      assert(false && "Unsupported dynamic accessor target");
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_dynamic_work_group_memory: {

    // We need to recover the inheritance layout by casting to
    // dynamic_parameter_impl first. Casting directly to
    // dynamic_work_group_memory_impl would result in an incorrect pointer.
    auto *DynParamImpl = static_cast<
        ext::oneapi::experimental::detail::dynamic_parameter_impl *>(Ptr);

    registerDynamicParameter(DynParamImpl, Index + IndexShift);

    auto *DynWorkGroupImpl = static_cast<
        ext::oneapi::experimental::detail::dynamic_work_group_memory_impl *>(
        DynParamImpl);

    addArg(kernel_param_kind_t::kind_std_layout, nullptr,
           DynWorkGroupImpl->BufferSizeInBytes, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_work_group_memory: {
    addArg(kernel_param_kind_t::kind_std_layout, nullptr,
           static_cast<detail::work_group_memory_impl *>(Ptr)->buffer_size,
           Index + IndexShift);
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

void handler::setArgHelper(int ArgIndex, detail::work_group_memory_impl &Arg) {
  impl->MWorkGroupMemoryObjects.push_back(
      std::make_shared<detail::work_group_memory_impl>(Arg));
  addArg(detail::kernel_param_kind_t::kind_work_group_memory,
         impl->MWorkGroupMemoryObjects.back().get(), 0, ArgIndex);
}

void handler::setArgHelper(int ArgIndex, stream &&Str) {
  void *StoredArg = storePlainArg(Str);
  addArg(detail::kernel_param_kind_t::kind_stream, StoredArg, sizeof(stream),
         ArgIndex);
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
    char *LambdaPtr, detail::kernel_param_desc_t (*ParamDescGetter)(int),
    size_t NumKernelParams, bool IsESIMD) {
  size_t IndexShift = 0;
  impl->MArgs.reserve(MaxNumAdditionalArgs * NumKernelParams);

  for (size_t I = 0; I < NumKernelParams; ++I) {
    detail::kernel_param_desc_t ParamDesc = ParamDescGetter(I);
    void *Ptr = LambdaPtr + ParamDesc.offset;
    const detail::kernel_param_kind_t &Kind = ParamDesc.kind;
    const int &Size = ParamDesc.info;
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
    } else if (Kind == detail::kernel_param_kind_t::kind_dynamic_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      // Only local targets are supported for dynamic accessors.
      assert(static_cast<access::target>(Size & AccessTargetMask) ==
             access::target::local);

      ext::oneapi::experimental::detail::dynamic_parameter_base
          *DynamicParamBase = static_cast<
              ext::oneapi::experimental::detail::dynamic_parameter_base *>(Ptr);
      Ptr = detail::getSyclObjImpl(*DynamicParamBase).get();
    } else if (Kind ==
               detail::kernel_param_kind_t::kind_dynamic_work_group_memory) {
      ext::oneapi::experimental::detail::dynamic_parameter_base
          *DynamicParamBase = static_cast<
              ext::oneapi::experimental::detail::dynamic_parameter_base *>(Ptr);
      Ptr = detail::getSyclObjImpl(*DynamicParamBase).get();
    }

    processArg(Ptr, Kind, Size, I, IndexShift,
               /*IsKernelCreatedFromSource=*/false, IsESIMD);
  }
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// TODO: Those functions are not used anymore, remove it in the next
// ABI-breaking window.
void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, const std::vector<detail::kernel_param_desc_t> &ParamDescs,
    bool IsESIMD) {
  const bool IsKernelCreatedFromSource = false;
  size_t IndexShift = 0;
  impl->MArgs.reserve(MaxNumAdditionalArgs * ParamDescs.size());

  for (size_t I = 0; I < ParamDescs.size(); ++I) {
    void *Ptr = LambdaPtr + ParamDescs[I].offset;
    const detail::kernel_param_kind_t &Kind = ParamDescs[I].kind;
    const int &Size = ParamDescs[I].info;
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

void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs, bool IsESIMD) {
  std::vector<detail::kernel_param_desc_t> ParamDescs(
      KernelArgs, KernelArgs + KernelArgsNum);
  extractArgsAndReqsFromLambda(LambdaPtr, ParamDescs, IsESIMD);
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

// Calling methods of kernel_impl requires knowledge of class layout.
// As this is impossible in header, there's a function that calls necessary
// method inside the library and returns the result.
detail::ABINeutralKernelNameStrT handler::getKernelName() {
  return MKernel->getName();
}

void handler::verifyUsedKernelBundleInternal(detail::string_view KernelName) {
  detail::kernel_bundle_impl *UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundlePtr(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  if (!impl->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  if (!UsedKernelBundleImplPtr->has_kernel(
          KernelID, detail::createSyclObjFromImpl<device>(impl->get_device())))
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
  impl->MAdvice = static_cast<ur_usm_advice_flags_t>(Advice);
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

// Simple host to device copy
void handler::ext_oneapi_copy(
    const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE);

  setType(detail::CGType::CopyImage);
}

// Host to device copy with offsets and extent
void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE, SrcOffset,
                         SrcExtent, DestOffset, {0, 0, 0}, CopyExtent);

  setType(detail::CGType::CopyImage);
}

// Simple device to host copy
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = Dest;

  detail::fill_copy_args(get_impl(), SrcImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST);

  setType(detail::CGType::CopyImage);
}

// Device to host copy with offsets and extent
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = Dest;

  detail::fill_copy_args(get_impl(), SrcImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST, SrcOffset,
                         {0, 0, 0}, DestOffset, DestExtent, CopyExtent);

  setType(detail::CGType::CopyImage);
}

// Simple HtoD or DtoH copy with USM device memory
void handler::ext_oneapi_copy(
    const void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &Desc,
    size_t DeviceRowPitch) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  ur_exp_image_copy_flags_t ImageCopyFlags = detail::getUrImageCopyFlags(
      get_pointer_type(Src,
                       createSyclObjFromImpl<context>(impl->get_context())),
      get_pointer_type(Dest,
                       createSyclObjFromImpl<context>(impl->get_context())));

  if (ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE ||
      ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
    detail::fill_copy_args(get_impl(), Desc, ImageCopyFlags, DeviceRowPitch,
                           DeviceRowPitch);
  } else {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: This copy function only performs host "
                          "to device or device to host copies!");
  }

  setType(detail::CGType::CopyImage);
}

// HtoD or DtoH copy with USM device memory, using offsets, extent
void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  ur_exp_image_copy_flags_t ImageCopyFlags = detail::getUrImageCopyFlags(
      get_pointer_type(Src,
                       createSyclObjFromImpl<context>(impl->get_context())),
      get_pointer_type(Dest,
                       createSyclObjFromImpl<context>(impl->get_context())));

  // Fill the host extent based on the type of copy.
  if (ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
    detail::fill_copy_args(get_impl(), DeviceImgDesc, ImageCopyFlags,
                           DeviceRowPitch, DeviceRowPitch, SrcOffset,
                           HostExtent, DestOffset, {0, 0, 0}, CopyExtent);
  } else if (ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
    detail::fill_copy_args(get_impl(), DeviceImgDesc, ImageCopyFlags,
                           DeviceRowPitch, DeviceRowPitch, SrcOffset, {0, 0, 0},
                           DestOffset, HostExtent, CopyExtent);
  } else {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: This copy function only performs host "
                          "to device or device to host copies!");
  }

  setType(detail::CGType::CopyImage);
}

// Simple device to device copy
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE);

  setType(detail::CGType::CopyImage);
}

// Device to device copy with offsets and extent
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE, SrcOffset,
                         {0, 0, 0}, DestOffset, {0, 0, 0}, CopyExtent);

  setType(detail::CGType::CopyImage);
}

// device to device image_mem_handle to usm
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    size_t DestRowPitch) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = Dest;

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE, 0,
                         DestRowPitch);

  setType(detail::CGType::CopyImage);
}

// device to device image_mem_handle to usm sub copy
void handler::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    size_t DestRowPitch, sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = reinterpret_cast<void *>(Src.raw_handle);
  MDstPtr = Dest;

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE, 0,
                         DestRowPitch, SrcOffset, {0, 0, 0}, DestOffset,
                         {0, 0, 0}, CopyExtent);

  setType(detail::CGType::CopyImage);
}

// device to device usm to image_mem_handle
void handler::ext_oneapi_copy(
    const void *Src,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    size_t SrcRowPitch, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE, SrcRowPitch,
                         0);

  setType(detail::CGType::CopyImage);
}

// device to device usm to image_mem_handle sub copy
void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    size_t SrcRowPitch, ext::oneapi::experimental::image_mem_handle Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = reinterpret_cast<void *>(Dest.raw_handle);

  detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc,
                         UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE, SrcRowPitch,
                         0, SrcOffset, {0, 0, 0}, DestOffset, {0, 0, 0},
                         CopyExtent);

  setType(detail::CGType::CopyImage);
}

// Simple DtoD or HtoH USM to USM copy
void handler::ext_oneapi_copy(
    const void *Src,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    size_t SrcRowPitch, void *Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    size_t DestRowPitch) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  ur_exp_image_copy_flags_t ImageCopyFlags = detail::getUrImageCopyFlags(
      get_pointer_type(Src,
                       createSyclObjFromImpl<context>(impl->get_context())),
      get_pointer_type(Dest,
                       createSyclObjFromImpl<context>(impl->get_context())));

  if (ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE ||
      ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_HOST) {
    detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc, ImageCopyFlags,
                           SrcRowPitch, DestRowPitch);
  } else {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: This copy function only performs device "
                          "to device or host to host copies!");
  }

  setType(detail::CGType::CopyImage);
}

// DtoD or HtoH USM to USM copy with offsets and extent
void handler::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    size_t SrcRowPitch, void *Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    size_t DestRowPitch, sycl::range<3> CopyExtent) {
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;

  ur_exp_image_copy_flags_t ImageCopyFlags = detail::getUrImageCopyFlags(
      get_pointer_type(Src,
                       createSyclObjFromImpl<context>(impl->get_context())),
      get_pointer_type(Dest,
                       createSyclObjFromImpl<context>(impl->get_context())));

  if (ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE ||
      ImageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_HOST) {
    detail::fill_copy_args(get_impl(), SrcImgDesc, DestImgDesc, ImageCopyFlags,
                           SrcRowPitch, DestRowPitch, SrcOffset, {0, 0, 0},
                           DestOffset, {0, 0, 0}, CopyExtent);
  } else {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Copy Error: This copy function only performs device "
                          "to device or host to host copies!");
  }

  setType(detail::CGType::CopyImage);
}

void handler::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::external_semaphore ExtSemaphore) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();

  switch (ExtSemaphore.handle_type) {
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      opaque_fd:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      win32_nt_handle:
    break;
  default:
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore used needs a user passed wait value.");
    break;
  }

  impl->MExternalSemaphore =
      (ur_exp_external_semaphore_handle_t)ExtSemaphore.raw_handle;
  impl->MWaitValue = {};
  setType(detail::CGType::SemaphoreWait);
}

void handler::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::external_semaphore ExtSemaphore,
    uint64_t WaitValue) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();

  switch (ExtSemaphore.handle_type) {
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      win32_nt_dx12_fence:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      timeline_fd:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      timeline_win32_nt_handle:
    break;
  default:
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore does not support user passed wait values.");
    break;
  }

  impl->MExternalSemaphore =
      (ur_exp_external_semaphore_handle_t)ExtSemaphore.raw_handle;
  impl->MWaitValue = WaitValue;
  setType(detail::CGType::SemaphoreWait);
}

void handler::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::external_semaphore ExtSemaphore) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();

  switch (ExtSemaphore.handle_type) {
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      opaque_fd:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      win32_nt_handle:
    break;
  default:
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore used needs a user passed signal value.");
    break;
  }

  impl->MExternalSemaphore =
      (ur_exp_external_semaphore_handle_t)ExtSemaphore.raw_handle;
  impl->MSignalValue = {};
  setType(detail::CGType::SemaphoreSignal);
}

void handler::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::external_semaphore ExtSemaphore,
    uint64_t SignalValue) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_bindless_images>();

  switch (ExtSemaphore.handle_type) {
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      win32_nt_dx12_fence:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      timeline_fd:
  case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
      timeline_win32_nt_handle:
    break;
  default:
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Invalid type of semaphore for this operation. The "
        "type of semaphore does not support user passed signal values.");
    break;
  }

  impl->MExternalSemaphore =
      (ur_exp_external_semaphore_handle_t)ExtSemaphore.raw_handle;
  impl->MSignalValue = SignalValue;
  setType(detail::CGType::SemaphoreSignal);
}

void handler::use_kernel_bundle(
    const kernel_bundle<bundle_state::executable> &ExecBundle) {

  if (&impl->get_context() !=
      detail::getSyclObjImpl(ExecBundle.get_context()).get())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the primary queue is different from the "
        "context associated with the kernel bundle");

  if (impl->MSubmissionSecondaryQueue &&
      impl->MSubmissionSecondaryQueue->get_context() !=
          ExecBundle.get_context())
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

  // Async alloc calls adapter immediately. Any explicit/implicit dependencies
  // are handled at that point, including in order queue deps. Further calls to
  // depends_on after an async alloc are explicitly disallowed.
  if (getType() == CGType::AsyncAlloc) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Cannot submit a dependency after an asynchronous "
                          "allocation has already been executed!");
  }

  auto EventGraph = EventImpl->getCommandGraph();
  queue_impl *Queue = impl->get_queue_or_null();
  if (Queue && EventGraph) {
    auto QueueGraph = Queue->getCommandGraph();

    if (EventGraph->getContextImplPtr().get() != &impl->get_context()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different context.");
    }

    if (&EventGraph->getDeviceImpl() != &impl->get_device()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different device.");
    }

    if (QueueGraph && QueueGraph != EventGraph) {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Cannot submit to a recording queue with a "
                            "dependency from a different graph.");
    }

    // If the event dependency has a graph, that means that the queue that
    // created it was in recording mode. If the current queue is not recording,
    // we need to set it to recording (implements the transitive queue recording
    // feature).
    if (!QueueGraph) {
      EventGraph->beginRecording(*Queue);
    }
  }

  if (auto Graph = getCommandGraph(); Graph) {
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

static bool checkContextSupports(detail::context_impl &ContextImpl,
                                 ur_context_info_t InfoQuery) {
  auto &Adapter = ContextImpl.getAdapter();
  ur_bool_t SupportsOp = false;
  Adapter->call<UrApiKind::urContextGetInfo>(ContextImpl.getHandleRef(),
                                             InfoQuery, sizeof(ur_bool_t),
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
  const bool supported = impl->get_device().supportsForwardProgress(
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
  if (impl->get_graph_or_null())
    return true;

  return checkContextSupports(impl->get_context(),
                              UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
}

bool handler::supportsUSMFill2D() {
  if (impl->get_graph_or_null())
    return true;

  return checkContextSupports(impl->get_context(),
                              UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
}

bool handler::supportsUSMMemset2D() {
  // memset use the same UR check as fill2D.
  return supportsUSMFill2D();
}

id<2> handler::computeFallbackKernelBounds(size_t Width, size_t Height) {
  device_impl &Dev = impl->get_device();
  range<2> ItemLimit = Dev.get_info<info::device::max_work_item_sizes<2>>() *
                       Dev.get_info<info::device::max_compute_units>();
  return id<2>{std::min(ItemLimit[0], Height), std::min(ItemLimit[1], Width)};
}

// TODO: do we need this still?
backend handler::getDeviceBackend() const {
  return impl->get_device().getBackend();
}

void handler::ext_intel_read_host_pipe(detail::string_view Name, void *Ptr,
                                       size_t Size, bool Block) {
  impl->HostPipeName = std::string_view(Name);
  impl->HostPipePtr = Ptr;
  impl->HostPipeTypeSize = Size;
  impl->HostPipeBlocking = Block;
  impl->HostPipeRead = 1;
  setType(detail::CGType::ReadWriteHostPipe);
}

void handler::ext_intel_write_host_pipe(detail::string_view Name, void *Ptr,
                                        size_t Size, bool Block) {
  impl->HostPipeName = std::string_view(Name);
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
  host_task([=, &Dev = impl->get_device(),
             WeakContextImpl = impl->get_context().weak_from_this()] {
    // Capture context as weak to avoid keeping it alive for too long. If it is
    // dead by the time this executes, the operation would not have been visible
    // anyway. Devices are alive till library shutdown so capturing a reference
    // to one is fine.
    if (std::shared_ptr<detail::context_impl> ContextImpl =
            WeakContextImpl.lock())
      ContextImpl->memcpyToHostOnlyDeviceGlobal(
          Dev, DeviceGlobalPtr, Src, DeviceGlobalTSize, IsDeviceImageScoped,
          NumBytes, Offset);
  });
}

void handler::memcpyFromHostOnlyDeviceGlobal(void *Dest,
                                             const void *DeviceGlobalPtr,
                                             bool IsDeviceImageScoped,
                                             size_t NumBytes, size_t Offset) {
  host_task([=, Context = impl->get_context().shared_from_this(),
             &Dev = impl->get_device()] {
    // Unlike memcpy to device_global, we need to keep the context alive in the
    // capture of this operation as we must be able to correctly copy the value
    // to the user-specified pointer. Device is guaranteed to live until SYCL RT
    // library shutdown (but even if it wasn't, alive conext has to guarantee
    // alive device).
    Context->memcpyFromHostOnlyDeviceGlobal(
        Dev, Dest, DeviceGlobalPtr, IsDeviceImageScoped, NumBytes, Offset);
  });
}

const std::shared_ptr<detail::context_impl> &
handler::getContextImplPtr() const {
  if (auto *Graph = impl->get_graph_or_null()) {
    return Graph->getContextImplPtr();
  }
  return impl->get_queue().getContextImplPtr();
}

detail::context_impl &handler::getContextImpl() const {
  if (auto *Graph = impl->get_graph_or_null()) {
    return *Graph->getContextImplPtr();
  }
  return impl->get_queue().getContextImpl();
}

void handler::setKernelCacheConfig(handler::StableKernelCacheConfig Config) {
  switch (Config) {
  case handler::StableKernelCacheConfig::Default:
    impl->MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;
    break;
  case handler::StableKernelCacheConfig::LargeSLM:
    impl->MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_LARGE_SLM;
    break;
  case handler::StableKernelCacheConfig::LargeData:
    impl->MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_LARGE_DATA;
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

void handler::setKernelWorkGroupMem(size_t Size) {
  throwIfGraphAssociated<syclex::detail::UnsupportedGraphFeatures::
                             sycl_ext_oneapi_work_group_scratch_memory>();
  impl->MKernelWorkGroupMemorySize = Size;
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
  if (auto *Graph = impl->get_graph_or_null()) {
    return Graph->shared_from_this();
  }

  return impl->get_queue().getCommandGraph();
}

void handler::setUserFacingNodeType(ext::oneapi::experimental::node_type Type) {
  impl->MUserFacingNodeType = Type;
}

kernel_bundle<bundle_state::input> handler::getKernelBundle() const {
  detail::kernel_bundle_impl *KernelBundleImplPtr =
      getOrInsertHandlerKernelBundlePtr(/*Insert=*/true);

  return detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
      *KernelBundleImplPtr);
}

std::optional<std::array<size_t, 3>> handler::getMaxWorkGroups() {
  device_impl &DeviceImpl = impl->get_device();
  std::array<size_t, 3> UrResult = {};
  auto Ret = DeviceImpl.getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
      DeviceImpl.getHandleRef(),
      UrInfoCode<
          ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
      sizeof(UrResult), &UrResult, nullptr);
  if (Ret == UR_RESULT_SUCCESS) {
    return UrResult;
  }
  return {};
}

std::tuple<std::array<size_t, 3>, bool> handler::getMaxWorkGroups_v2() {
  auto ImmRess = getMaxWorkGroups();
  if (ImmRess)
    return {*ImmRess, true};
  return {std::array<size_t, 3>{0, 0, 0}, false};
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void handler::setNDRangeUsed(bool Value) { (void)Value; }
#endif

void handler::registerDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_impl *DynamicParamImpl,
    int ArgIndex) {

  if (queue_impl *Queue = impl->get_queue_or_null();
      Queue && Queue->hasCommandGraph()) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Dynamic Parameters cannot be used with Graph Queue recording.");
  }
  if (!impl->get_graph_or_null()) {
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Dynamic Parameters cannot be used with normal SYCL submissions");
  }

  impl->MDynamicParameters.emplace_back(DynamicParamImpl, ArgIndex);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// TODO: Remove in the next ABI-breaking window.
void handler::registerDynamicParameter(
    ext::oneapi::experimental::detail::dynamic_parameter_base &DynamicParamBase,
    int ArgIndex) {
  ext::oneapi::experimental::detail::dynamic_parameter_impl *DynParamImpl =
      detail::getSyclObjImpl(DynamicParamBase).get();

  registerDynamicParameter(DynParamImpl, ArgIndex);
}
#endif

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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// TODO: This function is not used anymore, remove it in the next
// ABI-breaking window.
void handler::addAccessorReq(detail::AccessorImplPtr Accessor) {
  // Add accessor to the list of requirements.
  impl->CGData.MRequirements.push_back(Accessor.get());
  // Store copy of the accessor.
  impl->CGData.MAccStorage.push_back(std::move(Accessor));
}
#endif

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

void handler::setKernelNameBasedCachePtr(
    sycl::detail::KernelNameBasedCacheT *KernelNameBasedCachePtr) {
  impl->MKernelNameBasedCachePtr = KernelNameBasedCachePtr;
}

void handler::setKernelInfo(
    void *KernelFuncPtr, int KernelNumArgs,
    detail::kernel_param_desc_t (*KernelParamDescGetter)(int),
    bool KernelIsESIMD, bool KernelHasSpecialCaptures) {
  impl->MKernelFuncPtr = KernelFuncPtr;
  impl->MKernelNumArgs = KernelNumArgs;
  impl->MKernelParamDescGetter = KernelParamDescGetter;
  impl->MKernelIsESIMD = KernelIsESIMD;
  impl->MKernelHasSpecialCaptures = KernelHasSpecialCaptures;
}

void handler::instantiateKernelOnHost(void *InstantiateKernelOnHostPtr) {
  // Passing the pointer to the runtime is enough to prevent optimization.
  // We don't need to use the pointer for anything.
  (void)InstantiateKernelOnHostPtr;
}

void handler::saveCodeLoc(detail::code_location CodeLoc, bool IsTopCodeLoc) {
  MCodeLoc = CodeLoc;
  impl->MIsTopCodeLoc = IsTopCodeLoc;
}
void handler::saveCodeLoc(detail::code_location CodeLoc) {
  MCodeLoc = CodeLoc;
  impl->MIsTopCodeLoc = true;
}
void handler::copyCodeLoc(const handler &other) {
  MCodeLoc = other.MCodeLoc;
  impl->MIsTopCodeLoc = other.impl->MIsTopCodeLoc;
}

queue handler::getQueue() {
  return createSyclObjFromImpl<queue>(impl->get_queue());
}
namespace detail {
__SYCL_EXPORT void HandlerAccess::preProcess(handler &CGH,
                                             type_erased_cgfo_ty F) {
  queue_impl &Q = CGH.impl->get_queue();
  bool EventNeeded = !Q.isInOrder();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  handler_impl HandlerImpl{Q, nullptr, EventNeeded};
  handler AuxHandler{HandlerImpl};
#else
  handler AuxHandler{Q.shared_from_this(), EventNeeded};
#endif
  AuxHandler.copyCodeLoc(CGH);
  F(AuxHandler);
  auto E = AuxHandler.finalize();
  assert(!CGH.MIsFinalized &&
         "Can't do pre-processing if the command has been enqueued already!");
  if (EventNeeded)
    CGH.depends_on(E);
}
__SYCL_EXPORT void HandlerAccess::postProcess(handler &CGH,
                                              type_erased_cgfo_ty F) {
  // The "hacky" `handler`s manipulation mentioned near the declaration in
  // `handler.hpp` and implemented here is far from perfect. A better approach
  // would be
  //
  //    bool OrigNeedsEvent = CGH.needsEvent()
  //    assert(CGH.not_finalized/enqueued());
  //    if (!InOrderQueue)
  //      CGH.setNeedsEvent()
  //
  //    handler PostProcessHandler(Queue, OrigNeedsEvent)
  //    auto E = CGH.finalize(); // enqueue original or current last
  //                             // post-process
  //    if (!InOrder)
  //      PostProcessHandler.depends_on(E)
  //
  //    swap_impls(CGH, PostProcessHandler)
  //    return; // queue::submit finalizes PostProcessHandler and returns its
  //            // event if necessary.
  //
  // Still hackier than "real" `queue::submit` but at least somewhat sane.
  // That, however hasn't been tried yet and we have an even hackier approach
  // copied from what's been done in an old reductions implementation before
  // eventless submission work has started. Not sure how feasible the approach
  // above is at this moment.

  // This `finalize` is wrong (at least logically) if
  //   `assert(!CGH.eventNeeded())`
  auto E = CGH.finalize();
  queue_impl &Q = CGH.impl->get_queue();
  bool InOrder = Q.isInOrder();
  // Cannot use `CGH.eventNeeded()` alone as there might be subsequent
  // `postProcess` calls and we cannot address them properly similarly to the
  // `finalize` issue described above. `swap_impls` suggested above might be
  // able to handle this scenario naturally.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  handler_impl HandlerImpl{Q, nullptr, CGH.eventNeeded() || !InOrder};
  handler AuxHandler{HandlerImpl};
#else
  handler AuxHandler{Q.shared_from_this(), CGH.eventNeeded() || !InOrder};
#endif
  if (!InOrder)
    AuxHandler.depends_on(E);
  AuxHandler.copyCodeLoc(CGH);
  F(AuxHandler);
  CGH.MLastEvent = AuxHandler.finalize();
}
} // namespace detail
} // namespace _V1
} // namespace sycl
