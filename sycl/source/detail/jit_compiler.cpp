//==--- jit_compiler.cpp - SYCL runtime JIT compiler for kernel fusion -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/feature_test.hpp>
#if SYCL_EXT_JIT_ENABLE
#include <KernelFusion.h>
#include <detail/device_image_impl.hpp>
#include <detail/helpers.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/queue_impl.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/kernel_bundle.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::function<void(void *)> jit_compiler::CustomDeleterForLibHandle =
    [](void *StoredPtr) {
      if (!StoredPtr)
        return;
      std::ignore = sycl::detail::ur::unloadOsLibrary(StoredPtr);
    };

static inline void printPerformanceWarning(const std::string &Message) {
  if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() > 0) {
    std::cerr << "WARNING: " << Message << "\n";
  }
}

jit_compiler::jit_compiler()
    : LibraryHandle(nullptr, CustomDeleterForLibHandle) {
  auto checkJITLibrary = [this]() -> bool {
#ifdef _WIN32
    static const std::string dir = sycl::detail::OSUtil::getCurrentDSODir();
    static const std::string JITLibraryName = dir + "\\" + "sycl-jit.dll";
#else
    static const std::string JITLibraryName = "libsycl-jit.so";
#endif
    std::unique_ptr<void, decltype(CustomDeleterForLibHandle)> LibraryPtr(
        sycl::detail::ur::loadOsLibrary(JITLibraryName),
        CustomDeleterForLibHandle);
    if (LibraryPtr == nullptr) {
      printPerformanceWarning("Could not find JIT library " + JITLibraryName);
      return false;
    }

    this->AddToConfigHandle = reinterpret_cast<AddToConfigFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "addToJITConfiguration"));
    if (!this->AddToConfigHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->ResetConfigHandle = reinterpret_cast<ResetConfigFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "resetJITConfiguration"));
    if (!this->ResetConfigHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->FuseKernelsHandle = reinterpret_cast<FuseKernelsFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "fuseKernels"));
    if (!this->FuseKernelsHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->MaterializeSpecConstHandle =
        reinterpret_cast<MaterializeSpecConstFuncT>(
            sycl::detail::ur::getOsLibraryFuncAddress(
                LibraryPtr.get(), "materializeSpecConstants"));
    if (!this->MaterializeSpecConstHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->CalculateHashHandle = reinterpret_cast<CalculateHashFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "calculateHash"));
    if (!this->CalculateHashHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->CompileSYCLHandle = reinterpret_cast<CompileSYCLFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "compileSYCL"));
    if (!this->CompileSYCLHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    this->DestroyBinaryHandle = reinterpret_cast<DestroyBinaryFuncT>(
        sycl::detail::ur::getOsLibraryFuncAddress(LibraryPtr.get(),
                                                  "destroyBinary"));
    if (!this->DestroyBinaryHandle) {
      printPerformanceWarning(
          "Cannot resolve JIT library function entry point");
      return false;
    }

    LibraryHandle = std::move(LibraryPtr);
    return true;
  };
  Available = checkJITLibrary();
}

static ::jit_compiler::BinaryFormat
translateBinaryImageFormat(ur::DeviceBinaryType Type) {
  switch (Type) {
  case SYCL_DEVICE_BINARY_TYPE_SPIRV:
    return ::jit_compiler::BinaryFormat::SPIRV;
  case SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE:
    return ::jit_compiler::BinaryFormat::LLVM;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Format unsupported for JIT compiler");
  }
}

::jit_compiler::BinaryFormat getTargetFormat(const QueueImplPtr &Queue) {
  auto Backend = Queue->getDeviceImplPtr()->getBackend();
  switch (Backend) {
  case backend::ext_oneapi_level_zero:
  case backend::opencl:
    return ::jit_compiler::BinaryFormat::SPIRV;
  case backend::ext_oneapi_cuda:
    return ::jit_compiler::BinaryFormat::PTX;
  case backend::ext_oneapi_hip:
    return ::jit_compiler::BinaryFormat::AMDGCN;
  default:
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Backend unsupported by kernel fusion");
  }
}

::jit_compiler::TargetInfo getTargetInfo(const QueueImplPtr &Queue) {
  ::jit_compiler::BinaryFormat Format = getTargetFormat(Queue);
  return ::jit_compiler::TargetInfo::get(
      Format, static_cast<::jit_compiler::DeviceArchitecture>(
                  Queue->getDeviceImplPtr()->getDeviceArch()));
}

static ::jit_compiler::ParameterKind
translateArgType(kernel_param_kind_t Kind) {
  using PK = ::jit_compiler::ParameterKind;
  using kind = kernel_param_kind_t;
  switch (Kind) {
  case kind::kind_accessor:
    return PK::Accessor;
  case kind::kind_std_layout:
    return PK::StdLayout;
  case kind::kind_sampler:
    return PK::Sampler;
  case kind::kind_pointer:
    return PK::Pointer;
  case kind::kind_specialization_constants_buffer:
    return PK::SpecConstBuffer;
  case kind::kind_stream:
    return PK::Stream;
  case kind::kind_work_group_memory:
    return PK::WorkGroupMemory;
  case kind::kind_dynamic_work_group_memory:
    return PK::DynamicWorkGroupMemory;
  case kind::kind_invalid:
    return PK::Invalid;
  }
  return PK::Invalid;
}

enum class Promotion { None, Private, Local };

struct PromotionInformation {
  Promotion PromotionTarget;
  unsigned KernelIndex;
  unsigned ArgIndex;
  Requirement *Definition;
  NDRDescT NDRange;
  size_t LocalSize;
  size_t ElemSize;
  std::vector<bool> UsedParams;
};

using PromotionMap = std::unordered_map<SYCLMemObjI *, PromotionInformation>;

template <typename Obj> Promotion getPromotionTarget(const Obj &) {
  return Promotion::None;
}

static Promotion getInternalizationInfo(Requirement *Req) {
  auto AccPromotion = getPromotionTarget(Req->MPropertyList);

  auto *MemObj = static_cast<sycl::detail::SYCLMemObjT *>(Req->MSYCLMemObj);
  if (MemObj->getType() != SYCLMemObjI::MemObjType::Buffer) {
    // We currently do not support promotion on non-buffer memory objects (e.g.,
    // images).
    return Promotion::None;
  }
  Promotion BuffPromotion = getPromotionTarget(*MemObj);
  if (AccPromotion != Promotion::None && BuffPromotion != Promotion::None &&
      AccPromotion != BuffPromotion) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Contradicting promotion properties on accessor and "
                          "underlying buffer are not allowed");
  }
  return (AccPromotion != Promotion::None) ? AccPromotion : BuffPromotion;
}

static std::optional<size_t> getLocalSize(NDRDescT NDRange,
                                          std::optional<size_t> UserGlobalSize,
                                          Requirement *Req, Promotion Target) {
  assert((!UserGlobalSize.has_value() || Target != Promotion::Local) &&
         "Unexpected range rounding");
  auto NumElementsMem = static_cast<SYCLMemObjT *>(Req->MSYCLMemObj)->size();
  if (Target == Promotion::Private) {
    if (UserGlobalSize.has_value()) {
      // Only the first dimension is affected by range rounding.
      NDRange.GlobalSize[0] = *UserGlobalSize;
    }
    auto NumWorkItems = NDRange.GlobalSize.size();
    // For private internalization, the local size is
    // (Number of elements in buffer)/(number of work-items)
    return NumElementsMem / NumWorkItems;
  } else if (Target == Promotion::Local) {
    if (NDRange.LocalSize.size() == 0) {
      // No work-group size provided, cannot calculate the local size
      // and need to bail out.
      return {};
    }
    auto NumWorkGroups = NDRange.GlobalSize.size() / NDRange.LocalSize.size();
    // For local internalization, the local size is
    // (Number of elements in buffer)/(number of work-groups)
    return NumElementsMem / NumWorkGroups;
  }
  return 0;
}

static bool accessorEquals(Requirement *Req, Requirement *Other) {
  return Req->MOffset == Other->MOffset &&
         Req->MAccessRange == Other->MAccessRange &&
         Req->MMemoryRange == Other->MMemoryRange &&
         Req->MSYCLMemObj == Other->MSYCLMemObj && Req->MDims == Other->MDims &&
         Req->MElemSize == Other->MElemSize &&
         Req->MOffsetInBytes == Other->MOffsetInBytes &&
         Req->MIsSubBuffer == Other->MIsSubBuffer;
}

static void resolveInternalization(ArgDesc &Arg, unsigned KernelIndex,
                                   unsigned ArgFunctionIndex, NDRDescT NDRange,
                                   std::optional<size_t> UserGlobalSize,
                                   PromotionMap &Promotions) {
  assert(Arg.MType == kernel_param_kind_t::kind_accessor);

  Requirement *Req = static_cast<Requirement *>(Arg.MPtr);

  auto ThisPromotionTarget = getInternalizationInfo(Req);
  auto ThisLocalSize =
      getLocalSize(NDRange, UserGlobalSize, Req, ThisPromotionTarget);

  if (Promotions.count(Req->MSYCLMemObj)) {
    // We previously encountered an accessor for the same buffer.
    auto &PreviousDefinition = Promotions.at(Req->MSYCLMemObj);

    switch (ThisPromotionTarget) {
    case Promotion::None: {
      if (PreviousDefinition.PromotionTarget != Promotion::None) {
        printPerformanceWarning(
            "Deactivating previously specified promotion, because this "
            "accessor does not specify promotion");
        PreviousDefinition.PromotionTarget = Promotion::None;
      }
      return;
    }
    case Promotion::Local: {
      if (PreviousDefinition.PromotionTarget == Promotion::None) {
        printPerformanceWarning(
            "Not performing specified local promotion, due to previous "
            "mismatch or because previous accessor specified no promotion");
        return;
      }
      if (!ThisLocalSize.has_value()) {
        printPerformanceWarning("Work-group size for local promotion not "
                                "specified, not performing internalization");
        PreviousDefinition.PromotionTarget = Promotion::None;
        return;
      }
      if (PreviousDefinition.PromotionTarget == Promotion::Private) {
        printPerformanceWarning(
            "Overriding previous private promotion with local promotion");
        // Recompute the local size for the previous definition with adapted
        // promotion target.
        auto NewPrevLocalSize =
            getLocalSize(PreviousDefinition.NDRange, std::nullopt,
                         PreviousDefinition.Definition, Promotion::Local);

        if (!NewPrevLocalSize.has_value()) {
          printPerformanceWarning(
              "Not performing specified local promotion because previous "
              "kernels did not specify a local size");
          PreviousDefinition.PromotionTarget = Promotion::None;
          return;
        }

        PreviousDefinition.LocalSize = NewPrevLocalSize.value();
        PreviousDefinition.PromotionTarget = Promotion::Local;
      }
      if (PreviousDefinition.LocalSize != ThisLocalSize.value()) {
        printPerformanceWarning("Not performing specified local promotion due "
                                "to work-group size mismatch");
        PreviousDefinition.PromotionTarget = Promotion::None;
        return;
      }
      if (!accessorEquals(Req, PreviousDefinition.Definition)) {
        printPerformanceWarning("Not performing specified promotion, due to "
                                "accessor parameter mismatch");
        PreviousDefinition.PromotionTarget = Promotion::None;
        return;
      }
      return;
    }
    case Promotion::Private: {
      if (PreviousDefinition.PromotionTarget == Promotion::None) {
        printPerformanceWarning(
            "Not performing specified private promotion, due to previous "
            "mismatch or because previous accessor specified no promotion");
        return;
      }

      if (PreviousDefinition.PromotionTarget == Promotion::Local) {
        // Recompute the local size with adapted promotion target.
        auto ThisLocalSize =
            getLocalSize(NDRange, std::nullopt, Req, Promotion::Local);
        if (!ThisLocalSize.has_value()) {
          printPerformanceWarning("Work-group size for local promotion not "
                                  "specified, not performing internalization");
          PreviousDefinition.PromotionTarget = Promotion::None;
          return;
        }

        if (PreviousDefinition.LocalSize != ThisLocalSize.value()) {
          printPerformanceWarning(
              "Not performing specified local promotion due "
              "to work-group size mismatch");
          PreviousDefinition.PromotionTarget = Promotion::None;
          return;
        }

        if (!accessorEquals(Req, PreviousDefinition.Definition)) {
          printPerformanceWarning("Not performing local promotion, due to "
                                  "accessor parameter mismatch");
          PreviousDefinition.PromotionTarget = Promotion::None;
          return;
        }

        printPerformanceWarning(
            "Performing local internalization instead, because previous "
            "accessor specified local promotion");
        return;
      }

      // Previous accessors also specified private promotion.
      if (PreviousDefinition.LocalSize != ThisLocalSize.value()) {
        printPerformanceWarning(
            "Not performing specified private promotion due "
            "to work-group size mismatch");
        PreviousDefinition.PromotionTarget = Promotion::None;
        return;
      }
      if (!accessorEquals(Req, PreviousDefinition.Definition)) {
        printPerformanceWarning("Not performing specified promotion, due to "
                                "accessor parameter mismatch");
        PreviousDefinition.PromotionTarget = Promotion::None;
        return;
      }
      return;
    }
    }
  } else {
    if (ThisPromotionTarget == Promotion::Local && !ThisLocalSize.has_value()) {
      printPerformanceWarning("Work-group size for local promotion not "
                              "specified, not performing internalization");
      ThisPromotionTarget = Promotion::None;
      ThisLocalSize = 0;
    }
    assert(ThisLocalSize.has_value());
    Promotions.emplace(
        Req->MSYCLMemObj,
        PromotionInformation{ThisPromotionTarget, KernelIndex, ArgFunctionIndex,
                             Req, NDRange, ThisLocalSize.value(),
                             Req->MElemSize, std::vector<bool>()});
  }
}

// Identify a parameter by the argument description, the kernel index and the
// parameter index in that kernel.
struct Param {
  ArgDesc Arg;
  unsigned KernelIndex;
  unsigned ArgIndex;
  bool Used;
  Param(ArgDesc Argument, unsigned KernelIdx, unsigned ArgIdx, bool InUse)
      : Arg{Argument}, KernelIndex{KernelIdx}, ArgIndex{ArgIdx}, Used{InUse} {}
};

using ParamList = std::vector<Param>;

using ParamIterator = std::vector<Param>::iterator;

std::vector<Param>::const_iterator
detectIdenticalParameter(std::vector<Param> &Params, ArgDesc Arg) {
  for (auto I = Params.begin(); I < Params.end(); ++I) {
    // Two arguments of different type can never be identical.
    if (I->Arg.MType == Arg.MType) {
      if (Arg.MType == kernel_param_kind_t::kind_pointer ||
          Arg.MType == kernel_param_kind_t::kind_std_layout) {
        // Compare size and, if the size is identical, the content byte-by-byte.
        if ((Arg.MSize == I->Arg.MSize) &&
            std::memcmp(Arg.MPtr, I->Arg.MPtr, Arg.MSize) == 0) {
          return I;
        }
      } else if (Arg.MType == kernel_param_kind_t::kind_accessor) {
        Requirement *Req = static_cast<Requirement *>(Arg.MPtr);
        Requirement *Other = static_cast<Requirement *>(I->Arg.MPtr);
        if (accessorEquals(Req, Other)) {
          return I;
        }
      }
    }
  }
  return Params.end();
}

void *storePlainArgRaw(std::vector<std::vector<char>> &ArgStorage, void *ArgPtr,
                       size_t ArgSize) {
  ArgStorage.emplace_back(ArgSize);
  void *Storage = ArgStorage.back().data();
  std::memcpy(Storage, ArgPtr, ArgSize);
  return Storage;
}

template <typename T>
void *storePlainArg(std::vector<std::vector<char>> &ArgStorage, T &&Arg) {
  return storePlainArgRaw(ArgStorage, &Arg, sizeof(T));
}

static ParamIterator preProcessArguments(
    std::vector<std::vector<char>> &ArgStorage, ParamIterator Arg,
    PromotionMap &PromotedAccs,
    std::vector<::jit_compiler::ParameterInternalization> &InternalizeParams,
    std::vector<::jit_compiler::JITConstant> &JITConstants,
    ParamList &NonIdenticalParams,
    std::vector<::jit_compiler::ParameterIdentity> &ParamIdentities) {

  // Unused arguments are still in the list at this point (because we
  // need them for accessor handling), but there's not pre-processing
  // that needs to be done.
  if (!Arg->Used) {
    return ++Arg;
  }

  if (Arg->Arg.MType == kernel_param_kind_t::kind_pointer) {
    // Pointer arguments are only stored in the kernel functor object, which
    // will go out-of-scope before we execute the fused kernel. Therefore, we
    // need to copy the pointer (not the memory it's pointing to) to a permanent
    // location and update the argument.
    Arg->Arg.MPtr =
        storePlainArg(ArgStorage, *static_cast<void **>(Arg->Arg.MPtr));
  }
  if (Arg->Arg.MType == kernel_param_kind_t::kind_std_layout) {
    // Standard layout arguments are only stored in the kernel functor object,
    // which will go out-of-scope before we execute the fused kernel. Therefore,
    // we need to copy the argument to a permant location and update the
    // argument.
    if (Arg->Arg.MPtr) {
      Arg->Arg.MPtr =
          storePlainArgRaw(ArgStorage, Arg->Arg.MPtr, Arg->Arg.MSize);
      // Propagate values of scalar parameters as constants to the JIT
      // compiler.
      JITConstants.emplace_back(
          ::jit_compiler::Parameter{Arg->KernelIndex, Arg->ArgIndex},
          Arg->Arg.MPtr, Arg->Arg.MSize);
    }
    // Standard layout arguments do not participate in identical argument
    // detection, but we still add it to the list here. As the SYCL runtime can
    // only check the raw bytes for identical content, but is unaware of the
    // underlying datatype, some identities that would be detected here could
    // not be materialized by the JIT compiler. Instead of removing some
    // standard layout arguments due to identity and missing some in case the
    // materialization is not possible, we rely on constant propagation to
    // replace standard layout arguments by constants.
    NonIdenticalParams.emplace_back(Arg->Arg, Arg->KernelIndex, Arg->ArgIndex,
                                    true);
    return ++Arg;
  }
  // First check if there's already another parameter with identical
  // value.
  auto Identical = detectIdenticalParameter(NonIdenticalParams, Arg->Arg);
  if (Identical != NonIdenticalParams.end()) {
    ::jit_compiler::Parameter ThisParam{Arg->KernelIndex, Arg->ArgIndex};
    ::jit_compiler::Parameter IdenticalParam{Identical->KernelIndex,
                                             Identical->ArgIndex};
    ::jit_compiler::ParameterIdentity Identity{ThisParam, IdenticalParam};
    ParamIdentities.push_back(Identity);
    return ++Arg;
  }

  if (Arg->Arg.MType == kernel_param_kind_t::kind_accessor) {
    // Get local and private promotion information from accessors.
    Requirement *Req = static_cast<Requirement *>(Arg->Arg.MPtr);
    auto &Internalization = PromotedAccs.at(Req->MSYCLMemObj);
    auto PromotionTarget = Internalization.PromotionTarget;
    if (PromotionTarget == Promotion::Private ||
        PromotionTarget == Promotion::Local) {
      // The accessor should be promoted.
      if (Internalization.KernelIndex == Arg->KernelIndex &&
          Internalization.ArgIndex == Arg->ArgIndex) {
        // This is the first accessor for this buffer that should be
        // internalized.
        InternalizeParams.emplace_back(
            ::jit_compiler::Parameter{Arg->KernelIndex, Arg->ArgIndex},
            (PromotionTarget == Promotion::Private)
                ? ::jit_compiler::Internalization::Private
                : ::jit_compiler::Internalization::Local,
            Internalization.LocalSize, Internalization.ElemSize);
        // If an accessor will be promoted, i.e., if it has the promotion
        // property attached to it, the next three arguments, that are
        // associated with the accessor (access range, memory range, offset),
        // must not participate in identical parameter detection or constant
        // propagation, because their values will change if promotion happens.
        // Therefore, we can just skip them here, but we need to remember which
        // of them are used.
        for (unsigned I = 0; I < 4; ++I) {
          Internalization.UsedParams.push_back(Arg->Used);
          ++Arg;
        }
      } else {
        // We have previously encountered an accessor the same buffer, which
        // should be internalized. We can add parameter identities for the
        // accessor argument and the next three arguments (range, memory range
        // and offset, if they are used).
        unsigned Increment = 0;
        for (unsigned I = 0; I < 4; ++I) {
          // If the argument is used in both cases, i.e., on the original
          // accessor to be internalized, and this one, we can insert a
          // parameter identity.
          if (Arg->Used && Internalization.UsedParams[I]) {
            ::jit_compiler::Parameter ThisParam{Arg->KernelIndex,
                                                Arg->ArgIndex};
            ::jit_compiler::Parameter IdenticalParam{
                Internalization.KernelIndex,
                Internalization.ArgIndex + Increment};
            ::jit_compiler::ParameterIdentity Identity{ThisParam,
                                                       IdenticalParam};
            ParamIdentities.push_back(Identity);
          }
          if (Internalization.UsedParams[I]) {
            ++Increment;
          }
          ++Arg;
        }
      }
      return Arg;
    } else {
      // The accessor will not be promoted, so it can participate in identical
      // parameter detection.
      NonIdenticalParams.emplace_back(Arg->Arg, Arg->KernelIndex, Arg->ArgIndex,
                                      true);
      return ++Arg;
    }
  } else if (Arg->Arg.MType == kernel_param_kind_t::kind_pointer) {
    // No identical parameter exists, so add this to the list.
    NonIdenticalParams.emplace_back(Arg->Arg, Arg->KernelIndex, Arg->ArgIndex,
                                    true);
    return ++Arg;
  }
  return ++Arg;
}

static void
updatePromotedArgs(const ::jit_compiler::SYCLKernelInfo &FusedKernelInfo,
                   NDRDescT NDRange, std::vector<ArgDesc> &FusedArgs,
                   std::vector<std::vector<char>> &FusedArgStorage) {
  auto &ArgUsageInfo = FusedKernelInfo.Args.UsageMask;
  assert(ArgUsageInfo.size() == FusedArgs.size());
  for (size_t ArgIndex = 0; ArgIndex < ArgUsageInfo.size();) {
    bool PromotedToPrivate =
        (ArgUsageInfo[ArgIndex] & ::jit_compiler::ArgUsage::PromotedPrivate);
    bool PromotedToLocal =
        (ArgUsageInfo[ArgIndex] & ::jit_compiler::ArgUsage::PromotedLocal);
    if (PromotedToLocal || PromotedToPrivate) {
      // For each internalized accessor, we need to override four arguments
      // (see 'addArgsForGlobalAccessor' in handler.cpp for reference), i.e.,
      // the pointer itself, plus twice the range and the offset.
      auto &OldArgDesc = FusedArgs[ArgIndex];
      assert(OldArgDesc.MType == kernel_param_kind_t::kind_accessor);
      auto *Req = static_cast<Requirement *>(OldArgDesc.MPtr);

      // The stored args are all three-dimensional, but depending on the
      // actual number of dimensions of the accessor, only a part of that
      // argument is later on passed to the kernel.
      const size_t SizeAccField =
          sizeof(size_t) * (Req->MDims == 0 ? 1 : Req->MDims);
      // Compute the local size and use it for the range parameters (only
      // relevant for local promotion).
      size_t LocalSize = PromotedToLocal ? *getLocalSize(NDRange, std::nullopt,
                                                         Req, Promotion::Local)
                                         : 0;
      range<3> AccessRange{1, 1, LocalSize};
      void *RangeArg = storePlainArg(FusedArgStorage, AccessRange);
      // Use all-zero as the offset
      id<3> AcessOffset{0, 0, 0};
      void *OffsetArg = storePlainArg(FusedArgStorage, AcessOffset);

      // Override the arguments.
      // 1. Override the pointer with a std-layout argument with 'nullptr' as
      // value. handler.cpp does the same for local accessors.
      int SizeInBytes = Req->MElemSize * LocalSize;
      FusedArgs[ArgIndex] =
          ArgDesc{kernel_param_kind_t::kind_std_layout, nullptr, SizeInBytes,
                  static_cast<int>(ArgIndex)};
      ++ArgIndex;
      // 2. Access Range
      FusedArgs[ArgIndex] =
          ArgDesc{kernel_param_kind_t::kind_std_layout, RangeArg,
                  static_cast<int>(SizeAccField), static_cast<int>(ArgIndex)};
      ++ArgIndex;
      // 3. Memory Range
      FusedArgs[ArgIndex] =
          ArgDesc{kernel_param_kind_t::kind_std_layout, RangeArg,
                  static_cast<int>(SizeAccField), static_cast<int>(ArgIndex)};
      ++ArgIndex;
      // 4. Offset
      FusedArgs[ArgIndex] =
          ArgDesc{kernel_param_kind_t::kind_std_layout, OffsetArg,
                  static_cast<int>(SizeAccField), static_cast<int>(ArgIndex)};
      ++ArgIndex;
    } else {
      ++ArgIndex;
    }
  }
}

ur_kernel_handle_t jit_compiler::materializeSpecConstants(
    const QueueImplPtr &Queue, const RTDeviceBinaryImage *BinImage,
    const std::string &KernelName,
    const std::vector<unsigned char> &SpecConstBlob) {
#ifndef _WIN32
  if (!BinImage) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "No suitable IR available for materializing");
  }
  if (KernelName.empty()) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Cannot jit kernel with invalid kernel function name");
  }
  auto &PM = detail::ProgramManager::getInstance();
  if (auto CachedKernel =
          PM.getCachedMaterializedKernel(KernelName, SpecConstBlob))
    return CachedKernel;

  auto &RawDeviceImage = BinImage->getRawData();
  auto DeviceImageSize = static_cast<size_t>(RawDeviceImage.BinaryEnd -
                                             RawDeviceImage.BinaryStart);
  // Set 0 as the number of address bits, because the JIT compiler can set this
  // field based on information from LLVM module's data-layout.
  auto BinaryImageFormat = translateBinaryImageFormat(BinImage->getFormat());
  if (BinaryImageFormat == ::jit_compiler::BinaryFormat::INVALID) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "No suitable IR available for materializing");
  }
  ::jit_compiler::SYCLKernelBinaryInfo BinInfo{
      BinaryImageFormat, 0, RawDeviceImage.BinaryStart, DeviceImageSize};

  ::jit_compiler::TargetInfo TargetInfo = getTargetInfo(Queue);
  AddToConfigHandle(
      ::jit_compiler::option::JITTargetInfo::set(std::move(TargetInfo)));
  bool DebugEnabled =
      detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() > 0;
  AddToConfigHandle(
      ::jit_compiler::option::JITEnableVerbose::set(DebugEnabled));
  auto SetUpOption = [](const std::string &Value) {
    ::jit_compiler::JITEnvVar Option(Value.begin(), Value.end());
    return Option;
  };
  ::jit_compiler::JITEnvVar TargetCPUOpt = SetUpOption(
      detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_TARGET_CPU>::get());
  AddToConfigHandle(::jit_compiler::option::JITTargetCPU::set(TargetCPUOpt));
  ::jit_compiler::JITEnvVar TargetFeaturesOpt = SetUpOption(
      detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_TARGET_FEATURES>::get());
  AddToConfigHandle(
      ::jit_compiler::option::JITTargetFeatures::set(TargetFeaturesOpt));

  auto MaterializerResult =
      MaterializeSpecConstHandle(KernelName.c_str(), BinInfo, SpecConstBlob);
  if (MaterializerResult.failed()) {
    std::string Message{"Compilation for kernel failed with message:\n"};
    Message.append(MaterializerResult.getErrorMessage());
    if (DebugEnabled) {
      std::cerr << Message << "\n";
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), Message);
  }

  auto &MaterializerKernelInfo = MaterializerResult.getKernelInfo();
  sycl_device_binary_struct MaterializedRawDeviceImage{RawDeviceImage};
  MaterializedRawDeviceImage.BinaryStart =
      MaterializerKernelInfo.BinaryInfo.BinaryStart;
  MaterializedRawDeviceImage.BinaryEnd =
      MaterializerKernelInfo.BinaryInfo.BinaryStart +
      MaterializerKernelInfo.BinaryInfo.BinarySize;

  const bool OrigCacheCfg = SYCLConfig<SYCL_CACHE_IN_MEM>::get();
  if (OrigCacheCfg) {
    if (0 != setenv("SYCL_CACHE_IN_MEM", "0", true)) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Failed to set env variable in materialize spec constel.");
    }
    SYCLConfig<SYCL_CACHE_IN_MEM>::reset();
  }

  RTDeviceBinaryImage MaterializedRTDevBinImage{&MaterializedRawDeviceImage};
  const auto &Context = Queue->get_context();
  const auto &Device = Queue->get_device();
  auto NewKernel = PM.getOrCreateMaterializedKernel(
      MaterializedRTDevBinImage, Context, Device, KernelName, SpecConstBlob);

  if (OrigCacheCfg) {
    if (0 != setenv("SYCL_CACHE_IN_MEM", "1", true)) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Failed to set env variable in materialize spec const.");
    }
    SYCLConfig<SYCL_CACHE_IN_MEM>::reset();
  }

  return NewKernel;
#else  // _WIN32
  (void)Queue;
  (void)BinImage;
  (void)KernelName;
  (void)SpecConstBlob;
  return nullptr;
#endif // _WIN32
}

std::unique_ptr<detail::CG>
jit_compiler::fuseKernels(QueueImplPtr Queue,
                          std::vector<ExecCGCommand *> &InputKernels,
                          const property_list &) {
  if (!isAvailable()) {
    printPerformanceWarning("JIT library not available");
    return nullptr;
  }
  if (InputKernels.empty()) {
    printPerformanceWarning("Fusion list is empty");
    return nullptr;
  }

  // Retrieve the device binary from each of the input
  // kernels to hand them over to the JIT compiler.
  std::vector<::jit_compiler::SYCLKernelInfo> InputKernelInfo;
  std::vector<std::string> InputKernelNames;
  // Collect argument information from all input kernels.

  detail::CG::StorageInitHelper CGData;
  std::vector<std::vector<char>> &ArgsStorage = CGData.MArgsStorage;
  std::vector<detail::AccessorImplPtr> &AccStorage = CGData.MAccStorage;
  std::vector<Requirement *> &Requirements = CGData.MRequirements;
  std::vector<detail::EventImplPtr> &Events = CGData.MEvents;
  std::vector<::jit_compiler::NDRange> Ranges;
  ur_kernel_cache_config_t KernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;
  unsigned KernelIndex = 0;
  ParamList FusedParams;
  PromotionMap PromotedAccs;
  // TODO: Collect information about streams and figure out how
  // to fuse them.
  for (auto &RawCmd : InputKernels) {
    auto *KernelCmd = static_cast<ExecCGCommand *>(RawCmd);
    auto &CG = KernelCmd->getCG();
    assert(KernelCmd->isFusable());
    auto *KernelCG = static_cast<CGExecKernel *>(&CG);

    auto &KernelName = KernelCG->MKernelName;
    if (KernelName.empty()) {
      printPerformanceWarning(
          "Cannot fuse kernel with invalid kernel function name");
      return nullptr;
    }

    auto [DeviceImage, Program] =
        retrieveKernelBinary(Queue, KernelName.c_str(), KernelCG);
    if (!DeviceImage || !Program) {
      printPerformanceWarning("No suitable IR available for fusion");
      return nullptr;
    }
    const KernelArgMask *EliminatedArgs = nullptr;
    if (Program && (KernelCG->MSyclKernel == nullptr ||
                    !KernelCG->MSyclKernel->isCreatedFromSource())) {
      EliminatedArgs =
          detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
              Program, KernelName);
    }

    // Collect information about the arguments of this kernel.

    // Might need to sort the arguments in case they are not already sorted,
    // see also the similar code in commands.cpp.
    auto Args = KernelCG->MArgs;
    std::sort(Args.begin(), Args.end(), [](const ArgDesc &A, const ArgDesc &B) {
      return A.MIndex < B.MIndex;
    });

    // Determine whether the kernel has been subject to DPCPP's range rounding.
    // If so, the first argument will be the original ("user") range.
    std::optional<size_t> UserGlobalSize;
    if ((KernelName.find("_ZTSN4sycl3_V16detail18RoundedRangeKernel") == 0 ||
         KernelName.find("_ZTSN4sycl3_V16detail19__pf_kernel_wrapper") == 0) &&
        !Args.empty()) {
      auto &A0 = Args[0];
      [[maybe_unused]] auto Dims = KernelCG->MNDRDesc.Dims;
      assert(A0.MPtr && A0.MSize == static_cast<int>(Dims * sizeof(size_t)) &&
             A0.MType == kernel_param_kind_t::kind_std_layout &&
             "Unexpected signature for rounded range kernel");

      size_t *UGS = reinterpret_cast<size_t *>(A0.MPtr);
      // Range-rounding only applies to the first dimension.
      assert(UGS[0] > KernelCG->MNDRDesc.GlobalSize[1]);
      assert(Dims < 2 || UGS[1] == KernelCG->MNDRDesc.GlobalSize[1]);
      assert(Dims < 3 || UGS[2] == KernelCG->MNDRDesc.GlobalSize[2]);
      UserGlobalSize = UGS[0];
    }

    ::jit_compiler::SYCLArgumentDescriptor ArgDescriptor{Args.size()};
    size_t ArgIndex = 0;
    // The kernel function in SPIR-V will only have the non-eliminated
    // arguments, so keep track of this "actual" argument index.
    unsigned ArgFunctionIndex = 0;
    auto KindIt = ArgDescriptor.Kinds.begin();
    auto UsageMaskIt = ArgDescriptor.UsageMask.begin();
    for (auto &Arg : Args) {
      *KindIt = translateArgType(Arg.MType);
      ++KindIt;

      // DPC++ internally uses 'true' to indicate that an argument has been
      // eliminated, while the JIT compiler uses 'true' to indicate an
      // argument is used. Translate this here.
      bool Eliminated = EliminatedArgs && !EliminatedArgs->empty() &&
                        (*EliminatedArgs)[ArgIndex++];
      *UsageMaskIt = !Eliminated;
      ++UsageMaskIt;

      // If the argument has not been eliminated, i.e., is still present on
      // the kernel function in LLVM-IR/SPIR-V, collect information about the
      // argument for performance optimizations in the JIT compiler.
      if (!Eliminated) {
        if (Arg.MType == kernel_param_kind_t::kind_accessor) {
          resolveInternalization(Arg, KernelIndex, ArgFunctionIndex,
                                 KernelCG->MNDRDesc, UserGlobalSize,
                                 PromotedAccs);
        }
        FusedParams.emplace_back(Arg, KernelIndex, ArgFunctionIndex, true);
        ++ArgFunctionIndex;
      } else {
        FusedParams.emplace_back(Arg, KernelIndex, 0, false);
      }
    }

    // TODO: Check for the correct kernel bundle state of the device image?
    auto &RawDeviceImage = DeviceImage->getRawData();
    auto DeviceImageSize = static_cast<size_t>(RawDeviceImage.BinaryEnd -
                                               RawDeviceImage.BinaryStart);
    // Set 0 as the number of address bits, because the JIT compiler can set
    // this field based on information from SPIR-V/LLVM module's data-layout.
    auto BinaryImageFormat =
        translateBinaryImageFormat(DeviceImage->getFormat());
    if (BinaryImageFormat == ::jit_compiler::BinaryFormat::INVALID) {
      printPerformanceWarning("No suitable IR available for fusion");
      return nullptr;
    }
    ::jit_compiler::SYCLKernelBinaryInfo BinInfo{
        BinaryImageFormat, 0, RawDeviceImage.BinaryStart, DeviceImageSize};

    constexpr auto SYCLTypeToIndices = [](auto Val) -> ::jit_compiler::Indices {
      return {Val.get(0), Val.get(1), Val.get(2)};
    };

    auto &CurrentNDR = KernelCG->MNDRDesc;
    const ::jit_compiler::NDRange JITCompilerNDR{
        static_cast<int>(CurrentNDR.Dims),
        SYCLTypeToIndices(CurrentNDR.GlobalSize),
        SYCLTypeToIndices(CurrentNDR.LocalSize),
        SYCLTypeToIndices(CurrentNDR.GlobalOffset)};

    Ranges.push_back(JITCompilerNDR);
    InputKernelInfo.emplace_back(KernelName.c_str(), ArgDescriptor,
                                 JITCompilerNDR, BinInfo);

    // Collect information for the fused kernel

    if (CurrentNDR.GlobalSize[0] == 0 && CurrentNDR.NumWorkGroups[0] != 0) {
      // Some overloads of parallel_for_work_group only specify the number of
      // work-groups, so this can be used to identify hierarchical parallel
      // kernels, which are not supported by fusion for now.
      printPerformanceWarning(
          "Cannot fuse kernel with hierarchical parallelism");
      return nullptr;
      // Not all overloads of parallel_for_work_group only specify the number of
      // work-groups, so the above mechanism might not detect all hierarchical
      // parallelism.
      // TODO: Find a more reliable way to detect hierarchical parallelism.
    }

    // We need to copy the storages here. The input CGs might be eliminated
    // before the fused kernel gets executed, so we need to copy the storages
    // here to make sure the arguments don't die on us before executing the
    // fused kernel.
    ArgsStorage.insert(ArgsStorage.end(), KernelCG->getArgsStorage().begin(),
                       KernelCG->getArgsStorage().end());
    AccStorage.insert(AccStorage.end(), KernelCG->getAccStorage().begin(),
                      KernelCG->getAccStorage().end());
    // TODO: Does the MSharedPtrStorage contain any information about actual
    // shared pointers beside the kernel bundle and handler impl? If yes, we
    // might need to copy it here.
    Requirements.insert(Requirements.end(), KernelCG->getRequirements().begin(),
                        KernelCG->getRequirements().end());
    Events.insert(Events.end(), KernelCG->getEvents().begin(),
                  KernelCG->getEvents().end());

    // If all kernels have the same cache config then use it for the merged
    // kernel, otherwise use default configuration.
    if (KernelIndex == 0) {
      KernelCacheConfig = KernelCG->MKernelCacheConfig;
    } else if (KernelCG->MKernelCacheConfig != KernelCacheConfig) {
      KernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;
    }

    ++KernelIndex;
  }

  // Pre-process the arguments, to detect identical parameters or arguments that
  // can be constant-propagated by the JIT compiler.
  std::vector<::jit_compiler::ParameterInternalization> InternalizeParams;
  std::vector<::jit_compiler::JITConstant> JITConstants;
  std::vector<::jit_compiler::ParameterIdentity> ParamIdentities;
  ParamList NonIdenticalParameters;
  for (auto UR = FusedParams.begin(); UR != FusedParams.end();) {
    UR = preProcessArguments(ArgsStorage, UR, PromotedAccs, InternalizeParams,
                             JITConstants, NonIdenticalParameters,
                             ParamIdentities);
  }

  // Retrieve barrier flags.
  ::jit_compiler::BarrierFlags BarrierFlags =
      ::jit_compiler::getLocalAndGlobalBarrierFlag();

  static size_t FusedKernelNameIndex = 0;
  auto FusedKernelName = "fused_" + std::to_string(FusedKernelNameIndex++);
  ResetConfigHandle();
  bool DebugEnabled =
      detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() > 0;
  AddToConfigHandle(
      ::jit_compiler::option::JITEnableVerbose::set(DebugEnabled));
  AddToConfigHandle(::jit_compiler::option::JITEnableCaching::set(
      detail::SYCLConfig<detail::SYCL_ENABLE_FUSION_CACHING>::get()));

  ::jit_compiler::TargetInfo TargetInfo = getTargetInfo(Queue);
  ::jit_compiler::BinaryFormat TargetFormat = TargetInfo.getFormat();
  AddToConfigHandle(
      ::jit_compiler::option::JITTargetInfo::set(std::move(TargetInfo)));

  auto FusionResult = FuseKernelsHandle(
      InputKernelInfo, FusedKernelName.c_str(), ParamIdentities, BarrierFlags,
      InternalizeParams, JITConstants);

  if (FusionResult.failed()) {
    if (DebugEnabled) {
      std::cerr
          << "ERROR: JIT compilation for kernel fusion failed with message:\n"
          << FusionResult.getErrorMessage() << "\n";
    }
    return nullptr;
  }

  auto &FusedKernelInfo = FusionResult.getKernelInfo();
  std::string FusedOrCachedKernelName{FusedKernelInfo.Name.c_str()};

  std::vector<ArgDesc> FusedArgs;
  int FusedArgIndex = 0;
  for (auto &Param : FusedParams) {
    // Add to the argument list of the fused kernel, but with the correct
    // new index in the fused kernel.
    auto &Arg = Param.Arg;
    FusedArgs.emplace_back(Arg.MType, Arg.MPtr, Arg.MSize, FusedArgIndex++);
  }

  // Update the kernel arguments for internalized accessors.
  const auto NDRDesc = [](const auto &ND) -> NDRDescT {
    constexpr auto ToSYCLType = [](const auto &Indices) -> sycl::range<3> {
      return {Indices[0], Indices[1], Indices[2]};
    };
    NDRDescT NDRDesc;
    NDRDesc.Dims = ND.getDimensions();
    NDRDesc.GlobalSize = ToSYCLType(ND.getGlobalSize());
    NDRDesc.LocalSize = ToSYCLType(ND.getLocalSize());
    NDRDesc.GlobalOffset = ToSYCLType(ND.getOffset());
    return NDRDesc;
  }(FusedKernelInfo.NDR);
  updatePromotedArgs(FusedKernelInfo, NDRDesc, FusedArgs, ArgsStorage);

  if (!FusionResult.cached()) {
    auto PIDeviceBinaries = createPIDeviceBinary(FusedKernelInfo, TargetFormat);
    detail::ProgramManager::getInstance().addImages(PIDeviceBinaries);
  } else {
    if (DebugEnabled) {
      std::cerr << "INFO: Re-using existing device binary for fused kernel\n";
    }
  }

  // Create a kernel bundle for the fused kernel.
  // Kernel bundles are stored in the CG as one of the "extended" members.
  auto FusedKernelId = detail::ProgramManager::getInstance().getSYCLKernelID(
      FusedOrCachedKernelName);

  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr;

  std::unique_ptr<detail::CG> FusedCG;
  FusedCG.reset(new detail::CGExecKernel(
      NDRDesc, nullptr, nullptr, std::move(KernelBundleImplPtr),
      std::move(CGData), std::move(FusedArgs),
      std::move(FusedOrCachedKernelName), {}, {}, CGType::Kernel,
      KernelCacheConfig, false /* KernelIsCooperative */,
      false /* KernelUsesClusterLaunch*/, 0 /* KernelWorkGroupMemorySize */));
  return FusedCG;
}

sycl_device_binaries jit_compiler::createPIDeviceBinary(
    const ::jit_compiler::SYCLKernelInfo &FusedKernelInfo,
    ::jit_compiler::BinaryFormat Format) {

  const char *TargetSpec = nullptr;
  sycl_device_binary_type BinFormat = SYCL_DEVICE_BINARY_TYPE_NATIVE;
  switch (Format) {
  case ::jit_compiler::BinaryFormat::PTX: {
    TargetSpec = __SYCL_DEVICE_BINARY_TARGET_NVPTX64;
    BinFormat = SYCL_DEVICE_BINARY_TYPE_NONE;
    break;
  }
  case ::jit_compiler::BinaryFormat::AMDGCN: {
    TargetSpec = __SYCL_DEVICE_BINARY_TARGET_AMDGCN;
    BinFormat = SYCL_DEVICE_BINARY_TYPE_NONE;
    break;
  }
  case ::jit_compiler::BinaryFormat::SPIRV: {
    TargetSpec = (FusedKernelInfo.BinaryInfo.AddressBits == 64)
                     ? __SYCL_DEVICE_BINARY_TARGET_SPIRV64
                     : __SYCL_DEVICE_BINARY_TARGET_SPIRV32;
    BinFormat = SYCL_DEVICE_BINARY_TYPE_SPIRV;
    break;
  }
  default:
    sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                    "Invalid output format");
  }

  std::string FusedKernelName{FusedKernelInfo.Name.c_str()};
  DeviceBinaryContainer Binary;

  // Create an offload entry for the fused kernel.
  // It seems to be OK to set zero for most of the information here, at least
  // that is the case for compiled SPIR-V binaries.
  OffloadEntryContainer Entry{FusedKernelName, nullptr, 0, 0, 0};
  Binary.addOffloadEntry(std::move(Entry));

  // Create a property entry for the argument usage mask for the fused kernel.
  auto ArgMask = encodeArgUsageMask(FusedKernelInfo.Args.UsageMask);
  PropertyContainer ArgMaskProp{
      FusedKernelName, ArgMask.data(), ArgMask.size(),
      sycl_property_type::SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  // Create a property set for the argument usage masks of all kernels
  // (currently only one).
  PropertySetContainer ArgMaskPropSet{
      __SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO};

  ArgMaskPropSet.addProperty(std::move(ArgMaskProp));

  Binary.addProperty(std::move(ArgMaskPropSet));

  if (Format == ::jit_compiler::BinaryFormat::PTX ||
      Format == ::jit_compiler::BinaryFormat::AMDGCN) {
    // Add a program metadata property with the reqd_work_group_size attribute.
    // See CUDA UR (ur_cuda.cpp) _ur_program::set_metadata for reference.
    auto ReqdWGS = std::find_if(
        FusedKernelInfo.Attributes.begin(), FusedKernelInfo.Attributes.end(),
        [](const ::jit_compiler::SYCLKernelAttribute &Attr) {
          return Attr.Kind == ::jit_compiler::SYCLKernelAttribute::AttrKind::
                                  ReqdWorkGroupSize;
        });
    if (ReqdWGS != FusedKernelInfo.Attributes.end()) {
      auto Encoded = encodeReqdWorkGroupSize(*ReqdWGS);
      std::stringstream PropName;
      PropName << FusedKernelInfo.Name.c_str();
      PropName << __SYCL_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE;
      PropertyContainer ReqdWorkGroupSizeProp{
          PropName.str(), Encoded.data(), Encoded.size(),
          sycl_property_type::SYCL_PROPERTY_TYPE_BYTE_ARRAY};
      PropertySetContainer ProgramMetadata{
          __SYCL_PROPERTY_SET_PROGRAM_METADATA};
      ProgramMetadata.addProperty(std::move(ReqdWorkGroupSizeProp));
      Binary.addProperty(std::move(ProgramMetadata));
    }
  }
  if (Format == ::jit_compiler::BinaryFormat::AMDGCN) {
    PropertyContainer NeedFinalization{
        __SYCL_PROGRAM_METADATA_TAG_NEED_FINALIZATION, 1};
    PropertySetContainer ProgramMetadata{__SYCL_PROPERTY_SET_PROGRAM_METADATA};
    ProgramMetadata.addProperty(std::move(NeedFinalization));
    Binary.addProperty(std::move(ProgramMetadata));
  }

  DeviceBinariesCollection Collection;
  Collection.addDeviceBinary(
      std::move(Binary), FusedKernelInfo.BinaryInfo.BinaryStart,
      FusedKernelInfo.BinaryInfo.BinarySize, TargetSpec, BinFormat);

  JITDeviceBinaries.push_back(std::move(Collection));
  return JITDeviceBinaries.back().getPIDeviceStruct();
}

sycl_device_binaries jit_compiler::createDeviceBinaries(
    const ::jit_compiler::RTCBundleInfo &BundleInfo,
    const std::string &Prefix) {
  auto Collection = std::make_unique<DeviceBinariesCollection>();

  for (const auto &DevImgInfo : BundleInfo.DevImgInfos) {
    DeviceBinaryContainer Binary;
    for (const auto &Symbol : DevImgInfo.SymbolTable) {
      // Create an offload entry for each kernel. We prepend a unique prefix to
      // support reusing the same name across multiple RTC requests. The actual
      // entrypoints remain unchanged.
      // It seems to be OK to set zero for most of the information here, at
      // least that is the case for compiled SPIR-V binaries.
      std::string PrefixedName = Prefix + Symbol.c_str();
      OffloadEntryContainer Entry{PrefixedName, /*Addr=*/nullptr, /*Size=*/0,
                                  /*Flags=*/0, /*Reserved=*/0};
      Binary.addOffloadEntry(std::move(Entry));
    }

    for (const auto &FPS : DevImgInfo.Properties) {
      bool IsDeviceGlobalsPropSet =
          FPS.Name == __SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS;
      PropertySetContainer PropSet{FPS.Name.c_str()};
      for (const auto &FPV : FPS.Values) {
        if (FPV.IsUIntValue) {
          PropSet.addProperty(
              PropertyContainer{FPV.Name.c_str(), FPV.UIntValue});
        } else {
          std::string PrefixedName =
              (IsDeviceGlobalsPropSet ? Prefix : "") + FPV.Name.c_str();
          PropSet.addProperty(PropertyContainer{
              PrefixedName.c_str(), FPV.Bytes.begin(), FPV.Bytes.size(),
              sycl_property_type::SYCL_PROPERTY_TYPE_BYTE_ARRAY});
        }
      }
      Binary.addProperty(std::move(PropSet));

      Binary.setCompileOptions(BundleInfo.CompileOptions.c_str());
    }

    Collection->addDeviceBinary(std::move(Binary),
                                DevImgInfo.BinaryInfo.BinaryStart,
                                DevImgInfo.BinaryInfo.BinarySize,
                                (DevImgInfo.BinaryInfo.AddressBits == 64)
                                    ? __SYCL_DEVICE_BINARY_TARGET_SPIRV64
                                    : __SYCL_DEVICE_BINARY_TARGET_SPIRV32,
                                SYCL_DEVICE_BINARY_TYPE_SPIRV);
  }

  sycl_device_binaries Binaries = Collection->getPIDeviceStruct();

  std::lock_guard<std::mutex> Guard{RTCDeviceBinariesMutex};
  RTCDeviceBinaries.emplace(Binaries, std::move(Collection));
  return Binaries;
}

void jit_compiler::destroyDeviceBinaries(sycl_device_binaries Binaries) {
  std::lock_guard<std::mutex> Guard{RTCDeviceBinariesMutex};
  for (uint16_t i = 0; i < Binaries->NumDeviceBinaries; ++i) {
    DestroyBinaryHandle(Binaries->DeviceBinaries[i].BinaryStart);
  }
  RTCDeviceBinaries.erase(Binaries);
}

std::vector<uint8_t> jit_compiler::encodeArgUsageMask(
    const ::jit_compiler::ArgUsageMask &Mask) const {
  // This must match the decoding logic in program_manager.cpp.
  constexpr uint64_t NBytesForSize = 8;
  constexpr uint64_t NBitsInElement = 8;
  uint64_t Size = static_cast<uint64_t>(Mask.size());
  // Round the size to the next multiple of 8
  uint64_t RoundedSize =
      ((Size + (NBitsInElement - 1)) & (~(NBitsInElement - 1)));
  std::vector<uint8_t> Encoded((RoundedSize / NBitsInElement) + NBytesForSize,
                               0u);
  // First encode the size of the actual mask
  for (size_t i = 0; i < NBytesForSize; ++i) {
    uint8_t Byte =
        static_cast<uint8_t>((RoundedSize >> i * NBitsInElement) & 0xFF);
    Encoded[i] = Byte;
  }
  // Encode the actual mask bit-wise
  for (size_t i = 0; i < Size; ++i) {
    // DPC++ internally uses 'true' to indicate that an argument has been
    // eliminated, while the JIT compiler uses 'true' to indicate an argument
    // is used. Translate this here.
    if (!(Mask[i] & ::jit_compiler::ArgUsage::Used)) {
      uint8_t &Byte = Encoded[NBytesForSize + (i / NBitsInElement)];
      Byte |= static_cast<uint8_t>((1 << (i % NBitsInElement)));
    }
  }
  return Encoded;
}

std::vector<uint8_t> jit_compiler::encodeReqdWorkGroupSize(
    const ::jit_compiler::SYCLKernelAttribute &Attr) const {
  assert(Attr.Kind ==
         ::jit_compiler::SYCLKernelAttribute::AttrKind::ReqdWorkGroupSize);
  size_t NumBytes = sizeof(uint64_t) + (Attr.Values.size() * sizeof(uint32_t));
  std::vector<uint8_t> Encoded(NumBytes, 0u);
  uint8_t *Ptr = Encoded.data();
  // Skip 64-bit wide size argument with value 0 at the start of the data.
  // See CUDA UR (ur_cuda.cpp) _ur_program::set_metadata for reference.
  Ptr += sizeof(uint64_t);
  for (const auto &Val : Attr.Values) {
    auto UVal = static_cast<uint32_t>(Val);
    std::memcpy(Ptr, &UVal, sizeof(uint32_t));
    Ptr += sizeof(uint32_t);
  }
  return Encoded;
}

std::pair<sycl_device_binaries, std::string> jit_compiler::compileSYCL(
    const std::string &CompilationID, const std::string &SYCLSource,
    const std::vector<std::pair<std::string, std::string>> &IncludePairs,
    const std::vector<std::string> &UserArgs, std::string *LogPtr) {
  auto appendToLog = [LogPtr](const char *Msg) {
    if (LogPtr) {
      LogPtr->append(Msg);
    }
  };

  std::string SYCLFileName = CompilationID + ".cpp";
  ::jit_compiler::InMemoryFile SourceFile{SYCLFileName.c_str(),
                                          SYCLSource.c_str()};

  std::vector<::jit_compiler::InMemoryFile> IncludeFilesView;
  IncludeFilesView.reserve(IncludePairs.size());
  std::transform(IncludePairs.begin(), IncludePairs.end(),
                 std::back_inserter(IncludeFilesView), [](const auto &Pair) {
                   return ::jit_compiler::InMemoryFile{Pair.first.c_str(),
                                                       Pair.second.c_str()};
                 });
  std::vector<const char *> UserArgsView;
  UserArgsView.reserve(UserArgs.size());
  std::transform(UserArgs.begin(), UserArgs.end(),
                 std::back_inserter(UserArgsView),
                 [](const auto &Arg) { return Arg.c_str(); });

  std::string CacheKey;
  std::vector<char> CachedIR;
  if (PersistentDeviceCodeCache::isEnabled()) {
    auto Result =
        CalculateHashHandle(SourceFile, IncludeFilesView, UserArgsView);

    if (Result.failed()) {
      appendToLog(Result.getPreprocLog());
    } else {
      CacheKey = Result.getHash();
      CachedIR = PersistentDeviceCodeCache::getDeviceCodeIRFromDisc(CacheKey);
    }
  }

  auto Result = CompileSYCLHandle(SourceFile, IncludeFilesView, UserArgsView,
                                  CachedIR, /*SaveIR=*/!CacheKey.empty());

  const char *BuildLog = Result.getBuildLog();
  appendToLog(BuildLog);
  switch (Result.getErrorCode()) {
    using RTCErrC = ::jit_compiler::RTCErrorCode;
  case RTCErrC::BUILD:
    throw sycl::exception(sycl::errc::build, BuildLog);
  case RTCErrC::INVALID:
    throw sycl::exception(sycl::errc::invalid, BuildLog);
  default: // RTCErrC::SUCCESS
    break;
  }

  const auto &IR = Result.getDeviceCodeIR();
  if (!CacheKey.empty() && !IR.empty()) {
    // The RTC result contains the bitcode blob iff the frontend was invoked on
    // the source string, meaning we encountered either a cache miss, or a cache
    // hit that returned unusable IR (e.g. due to a bitcode version mismatch).
    // There's no explicit mechanism to invalidate the cache entry - we just
    // overwrite the entry with the newly compiled IR.
    std::vector<char> SavedIR{IR.begin(), IR.end()};
    PersistentDeviceCodeCache::putDeviceCodeIRToDisc(CacheKey, SavedIR);
  }

  std::string Prefix = CompilationID + '$';
  return std::make_pair(createDeviceBinaries(Result.getBundleInfo(), Prefix),
                        std::move(Prefix));
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // SYCL_EXT_JIT_ENABLE
