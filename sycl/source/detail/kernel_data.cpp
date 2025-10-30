//==-------------------- kernel_data.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_data.hpp>

#include <detail/accessor_impl.hpp>

#include <sycl/accessor.hpp>
#include <sycl/stream.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

// The argument can take up more space to store additional information about
// MAccessRange, MMemoryRange, and MOffset added with addArgsForGlobalAccessor.
// We use the worst-case estimate because the lifetime of the vector is short.
// In processArg the kind_stream case introduces the maximum number of
// additional arguments. The case adds additional 12 arguments to the currently
// processed argument, hence worst-case estimate is 12+1=13.
// TODO: the constant can be removed if the size of MArgs will be calculated at
// compile time.
inline constexpr size_t MaxNumAdditionalArgs = 13;

constexpr static int AccessTargetMask = 0x7ff;

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

void KernelData::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                            const int Size, const size_t Index,
                            size_t &IndexShift, bool IsKernelCreatedFromSource
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
                            ,
                            bool IsESIMD
#endif
) {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  bool IsESIMD = isESIMD();
#endif
  using detail::kernel_param_kind_t;
  size_t GlobalSize = MNDRDesc.GlobalSize[0];
  for (size_t I = 1; I < MNDRDesc.Dims; ++I) {
    GlobalSize *= MNDRDesc.GlobalSize[I];
  }

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
    detail::Requirement *GBufReq = &*detail::getSyclObjImpl(*GBufBase);
    addArgsForGlobalAccessor(GBufReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, MArgs,
                             IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GOffsetBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalOffset);
    detail::Requirement *GOffsetReq = &*detail::getSyclObjImpl(*GOffsetBase);
    addArgsForGlobalAccessor(GOffsetReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, MArgs,
                             IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GFlushBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalFlushBuf);
    detail::Requirement *GFlushReq = &*detail::getSyclObjImpl(*GFlushBase);

    // If work group size wasn't set explicitly then it must be recieved
    // from kernel attribute or set to default values.
    // For now we can't get this attribute here.
    // So we just suppose that WG size is always default for stream.
    // TODO adjust MNDRDesc when device image contains kernel's attribute
    if (GlobalSize == 0) {
      GlobalSize = MNDRDesc.NumWorkGroups[0];
      for (size_t I = 1; I < MNDRDesc.Dims; ++I) {
        GlobalSize *= MNDRDesc.NumWorkGroups[I];
      }
    }
    addArgsForGlobalAccessor(GFlushReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, MArgs,
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
      addArgsForGlobalAccessor(AccImpl, Index, IndexShift, Size,
                               IsKernelCreatedFromSource, GlobalSize, MArgs,
                               IsESIMD);
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAccImpl =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      addArgsForLocalAccessor(LAccImpl, Index, IndexShift,
                              IsKernelCreatedFromSource, MArgs, IsESIMD);
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

      MDynamicParameters.emplace_back(DynParamImpl, Index + IndexShift);

      auto *DynLocalAccessorImpl = static_cast<
          ext::oneapi::experimental::detail::dynamic_local_accessor_impl *>(
          DynParamImpl);

      addArgsForLocalAccessor(&DynLocalAccessorImpl->LAccImplHost, Index,
                              IndexShift, IsKernelCreatedFromSource, MArgs,
                              IsESIMD);
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

    MDynamicParameters.emplace_back(DynParamImpl, Index + IndexShift);

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

void KernelData::extractArgsAndReqs(bool IsKernelCreatedFromSource) {
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
  clearArgs();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  MArgs.reserve(MaxNumAdditionalArgs * UnPreparedArgs.size());

  size_t IndexShift = 0;
  for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
    void *Ptr = UnPreparedArgs[I].MPtr;
    const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
    const int &Size = UnPreparedArgs[I].MSize;
    const int Index = UnPreparedArgs[I].MIndex;
    processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
               ,
               isESIMD()
#endif
    );
  }
}

void KernelData::extractArgsAndReqsFromLambda() {
  size_t IndexShift = 0;
  clearArgs();
  MArgs.reserve(MaxNumAdditionalArgs * getKernelNumArgs());

  for (size_t I = 0; I < getKernelNumArgs(); ++I) {
    auto KernelParamDescGetter = getKernelParamDescGetter();
    detail::kernel_param_desc_t ParamDesc = KernelParamDescGetter(I);
    void *Ptr = (char *)MKernelFuncPtr + ParamDesc.offset;
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
               /*IsKernelCreatedFromSource=*/false
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
               ,
               isESIMD()
#endif
    );
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
