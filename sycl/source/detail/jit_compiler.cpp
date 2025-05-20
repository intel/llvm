//==--- jit_compiler.cpp - SYCL runtime JIT compiler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/feature_test.hpp>
#if SYCL_EXT_JIT_ENABLE
#include <detail/jit_compiler.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/ur.hpp>

#include <llvm/Support/PropertySetIO.h>

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

#ifndef _WIN32
// These helpers are not Windows-specific, but they are only used by
// `materializeSpecConstants`, which isn't available on Windows.

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

static ::jit_compiler::BinaryFormat getTargetFormat(const QueueImplPtr &Queue) {
  auto Backend = Queue->getDeviceImpl().getBackend();
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
#endif // _WIN32

ur_kernel_handle_t jit_compiler::materializeSpecConstants(
    const QueueImplPtr &Queue, const RTDeviceBinaryImage *BinImage,
    KernelNameStrRefT KernelName,
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
  ::jit_compiler::JITBinaryInfo BinInfo{
      BinaryImageFormat, 0, RawDeviceImage.BinaryStart, DeviceImageSize};

  ::jit_compiler::BinaryFormat Format = getTargetFormat(Queue);

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

  auto MaterializerResult = MaterializeSpecConstHandle(
      KernelName.data(), BinInfo, Format, SpecConstBlob);
  if (MaterializerResult.failed()) {
    std::string Message{"Compilation for kernel failed with message:\n"};
    Message.append(MaterializerResult.getErrorMessage());
    if (DebugEnabled) {
      std::cerr << Message << "\n";
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), Message);
  }

  auto &MaterializerBinaryInfo = MaterializerResult.getBinaryInfo();
  sycl_device_binary_struct MaterializedRawDeviceImage{RawDeviceImage};
  MaterializedRawDeviceImage.BinaryStart = MaterializerBinaryInfo.BinaryStart;
  MaterializedRawDeviceImage.BinaryEnd =
      MaterializerBinaryInfo.BinaryStart + MaterializerBinaryInfo.BinarySize;

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
          FPS.Name == llvm::util::PropertySetRegistry::SYCL_DEVICE_GLOBALS;
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
    using RTCErrC = ::jit_compiler::RTCResult::RTCErrorCode;
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
