//==-------- handler.cpp --- SYCL command group handler --------*- C++ -*---==//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive property
// of Intel Corporation and may not be disclosed, examined or reproduced in
// whole or in part without explicit written authorization from the company.
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/kernel_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/info/info_desc.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
event handler::finalize() {
  sycl::event EventRet;
  unique_ptr_class<detail::CG> CommandGroup;
  switch (MCGType) {
  case detail::CG::KERNEL:
  case detail::CG::RUN_ON_HOST_INTEL: {
    shared_ptr_class<detail::kernel_impl> KernelImpl = nullptr;
    if (MSyclKernel)
      KernelImpl = detail::getSyclObjImpl(*MSyclKernel);
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(MNDRDesc), std::move(MHostKernel), std::move(KernelImpl),
        std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), std::move(MArgs), std::move(MKernelName),
        std::move(MOSModuleHandle), std::move(MStreamStorage), MCGType));
    break;
  }
  case detail::CG::COPY_ACC_TO_PTR:
  case detail::CG::COPY_PTR_TO_ACC:
  case detail::CG::COPY_ACC_TO_ACC:
    CommandGroup.reset(
        new detail::CGCopy(MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
                           std::move(MAccStorage), std::move(MSharedPtrStorage),
                           std::move(MRequirements), std::move(MEvents)));
    break;
  case detail::CG::FILL:
    CommandGroup.reset(new detail::CGFill(
        std::move(MPattern), MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents)));
    break;
  case detail::CG::UPDATE_HOST:
    CommandGroup.reset(new detail::CGUpdateHost(
        MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents)));
    break;
  case detail::CG::COPY_USM:
    CommandGroup.reset(new detail::CGCopyUSM(
        MSrcPtr, MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents)));
    break;
  case detail::CG::FILL_USM:
    CommandGroup.reset(new detail::CGFillUSM(
        std::move(MPattern), MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents)));
    break;
  case detail::CG::PREFETCH_USM:
    CommandGroup.reset(new detail::CGPrefetchUSM(
        MDstPtr, MLength, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents)));
    break;
  case detail::CG::NONE:
    throw runtime_error("Command group submitted without a kernel or a "
                        "explicit memory operation.");
  default:
    throw runtime_error("Unhandled type of command group");
  }

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  EventRet = detail::createSyclObjFromImpl<event>(Event);
  return EventRet;
}

void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource) {
  const auto kind_std_layout = detail::kernel_param_kind_t::kind_std_layout;
  const auto kind_accessor = detail::kernel_param_kind_t::kind_accessor;
  const auto kind_sampler = detail::kernel_param_kind_t::kind_sampler;
  const auto kind_pointer = detail::kernel_param_kind_t::kind_pointer;

  switch (Kind) {
  case kind_std_layout:
  case kind_pointer: {
    MArgs.emplace_back(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kind_accessor: {
    // For args kind of accessor Size is information about accessor.
    // The first 11 bits of Size encodes the accessor target.
    const access::target AccTarget = static_cast<access::target>(Size & 0x7ff);
    switch (AccTarget) {
    case access::target::global_buffer:
    case access::target::constant_buffer: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // Dimensionality of the buffer is 1 when dimensionality of the
        // accessor is 0.
        const size_t SizeAccField =
            sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
        ++IndexShift;
        MArgs.emplace_back(kind_std_layout, &AccImpl->MAccessRange[0],
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kind_std_layout, &AccImpl->MMemoryRange[0],
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kind_std_layout, &AccImpl->MOffset[0], SizeAccField,
                           Index + IndexShift);
      }
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAcc =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);
      // Stream implementation creates local accessor with size per work item
      // in work group. Number of work items is not available during stream
      // construction, that is why size of the accessor is updated here using
      // information about number of work items in the work group.
      if (LAcc->PerWI)
        LAcc->resize(MNDRDesc.LocalSize.size(), MNDRDesc.GlobalSize.size());
      range<3> &Size = LAcc->MSize;
      const int Dims = LAcc->MDims;
      int SizeInBytes = LAcc->MElemSize;
      for (int I = 0; I < Dims; ++I)
        SizeInBytes *= Size[I];
      MArgs.emplace_back(kind_std_layout, nullptr, SizeInBytes,
                         Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = Dims * sizeof(Size[0]);
        MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                           Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                           Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kind_std_layout, &Size, SizeAccField,
                           Index + IndexShift);
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
          "Unsupported accessor target case.");
      break;
    }
    }
    break;
  }
  case kind_sampler: {
    MArgs.emplace_back(kind_sampler, Ptr, sizeof(sampler), Index + IndexShift);
    break;
  }
  }
}

void handler::extractArgsAndReqs() {
  assert(MSyclKernel && "MSyclKernel is not initialized");
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
  MArgs.clear();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  const bool IsKernelCreatedFromSource =
      detail::getSyclObjImpl(*MSyclKernel)->isCreatedFromSource();

  size_t IndexShift = 0;
  for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
    void *Ptr = UnPreparedArgs[I].MPtr;
    const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
    const int &Size = UnPreparedArgs[I].MSize;
    const int Index = UnPreparedArgs[I].MIndex;
    processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource);
  }
}

void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs) {
  const bool IsKernelCreatedFromSource = false;
  size_t IndexShift = 0;
  for (size_t I = 0; I < KernelArgsNum; ++I) {
    void *Ptr = LambdaPtr + KernelArgs[I].offset;
    const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
    const int &Size = KernelArgs[I].info;
    if (Kind == detail::kernel_param_kind_t::kind_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & 0x7ff);
      if ((AccTarget == access::target::global_buffer ||
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
    processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource);
  }
}
} // namespace sycl
} // namespace cl
