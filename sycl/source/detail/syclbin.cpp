//==--------------------- syclbin.cpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/compiler.hpp>
#include <detail/device_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>

// SYCL_RT_HAS_LLVMOBJECT is defined by sycl/source/CMakeLists.txt for the
// runtime flavour whose CRT matches the LLVM build. The other flavour (for
// example, sycl9d when LLVM is built with the release CRT) compiles this TU
// with the macro undefined and falls back to throwing stubs so the runtime
// still links without referencing any LLVM symbols.
#ifdef SYCL_RT_HAS_LLVMOBJECT
#include "llvm/Object/SYCLBIN.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PropertySetIO.h"
#endif

#include <cstring>
#include <deque>
#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifdef SYCL_RT_HAS_LLVMOBJECT

namespace {

std::unique_ptr<char[]> copyContent(const char *Data, size_t Size) {
  std::unique_ptr<char[]> Result{new char[Size]};
  std::memcpy(Result.get(), Data, Size);
  return Result;
}

const char *getDeviceTargetSpecFromTriple(std::string_view Triple) {
  size_t TargetSize = Triple.find('-');
  if (TargetSize == Triple.npos)
    return __SYCL_DEVICE_BINARY_TARGET_UNKNOWN;
  std::string_view Target = Triple.substr(0, TargetSize);

  // Return the known targets to ensure null-terminated c-style strings.
  if (Target == __SYCL_DEVICE_BINARY_TARGET_UNKNOWN)
    return __SYCL_DEVICE_BINARY_TARGET_UNKNOWN;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_SPIRV32)
    return __SYCL_DEVICE_BINARY_TARGET_SPIRV32;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_SPIRV64)
    return __SYCL_DEVICE_BINARY_TARGET_SPIRV64;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64)
    return __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN)
    return __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NVPTX64)
    return __SYCL_DEVICE_BINARY_TARGET_NVPTX64;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_AMDGCN)
    return __SYCL_DEVICE_BINARY_TARGET_AMDGCN;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU)
    return __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU;
  return UR_DEVICE_BINARY_TARGET_UNKNOWN;
}

[[noreturn]] void throwInvalid(llvm::Error E) {
  std::string Msg = llvm::toString(std::move(E));
  throw sycl::exception(make_error_code(errc::invalid), std::move(Msg));
}

template <typename T> T expectOrThrow(llvm::Expected<T> &&E) {
  if (!E)
    throwInvalid(E.takeError());
  return std::move(*E);
}

} // namespace

// Opaque implementation. Holds all LLVM-typed state so no LLVM headers are
// pulled into runtime headers transitively.
struct SYCLBINBinaries::Impl {
  Impl(const char *Data, size_t Size)
      : ContentCopy{copyContent(Data, Size)}, ContentSize{Size} {
    llvm::MemoryBufferRef Buffer(
        llvm::StringRef(ContentCopy.get(), ContentSize), "");
    // SYCLBIN::read accepts the whole on-disk file and dispatches between
    // the v1 (legacy SYBI-magic) and v2 (multi-entry OffloadBinary) layouts
    // internally.
    ParsedSYCLBIN = expectOrThrow(llvm::object::SYCLBIN::read(Buffer));
    populateBinaries();
  }

  // Deferred buffers for sycl_device_binary_struct fields. Use std::deque so
  // pointers into the holders remain stable as new entries are appended.
  std::unique_ptr<char[]> ContentCopy;
  size_t ContentSize = 0;

  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN;

  std::deque<std::vector<_sycl_device_binary_property_struct>> BinaryProperties;
  std::deque<std::vector<_sycl_device_binary_property_set_struct>>
      BinaryPropertySets;

  std::vector<sycl_device_binary_struct> DeviceBinaries;

  struct AbstractModuleDesc {
    size_t NumJITBinaries = 0;
    size_t NumNativeBinaries = 0;
    RTDeviceBinaryImage *JITBinaries = nullptr;
    RTDeviceBinaryImage *NativeBinaries = nullptr;
  };
  std::unique_ptr<AbstractModuleDesc[]> AbstractModuleDescriptors;
  std::unique_ptr<RTDeviceBinaryImage[]> BinaryImages;

  uint8_t getState() const {
    auto &Global = (*ParsedSYCLBIN->GlobalMetadata)
        [llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA];
    auto It = Global.find(llvm::StringRef("state"));
    if (It == Global.end())
      throw sycl::exception(make_error_code(errc::invalid),
                            "SYCLBIN global metadata missing 'state'.");
    return static_cast<uint8_t>(It->second.asUint32());
  }

  size_t getNumAbstractModules() const {
    return ParsedSYCLBIN->AbstractModules.size();
  }

private:
  void populateBinaries();

  std::vector<_sycl_device_binary_property_set_struct> &
  convertAbstractModuleProperties(
      const llvm::object::SYCLBIN::AbstractModule &AM);
};

void SYCLBINBinaries::Impl::populateBinaries() {
  AbstractModuleDescriptors = std::unique_ptr<AbstractModuleDesc[]>(
      new AbstractModuleDesc[ParsedSYCLBIN->AbstractModules.size()]);

  size_t NumBinaries = 0;
  for (const auto &AM : ParsedSYCLBIN->AbstractModules)
    NumBinaries += AM.IRModules.size() + AM.NativeDeviceCodeImages.size();
  DeviceBinaries.reserve(NumBinaries);
  BinaryImages = std::unique_ptr<RTDeviceBinaryImage[]>(
      new RTDeviceBinaryImage[NumBinaries]);

  RTDeviceBinaryImage *CurrentBinaryImagesStart = BinaryImages.get();
  for (size_t I = 0; I < getNumAbstractModules(); ++I) {
    const auto &AM = ParsedSYCLBIN->AbstractModules[I];
    AbstractModuleDesc &AMDesc = AbstractModuleDescriptors[I];

    AMDesc.NumJITBinaries = AM.IRModules.size();
    AMDesc.NumNativeBinaries = AM.NativeDeviceCodeImages.size();
    AMDesc.JITBinaries = CurrentBinaryImagesStart;
    AMDesc.NativeBinaries = CurrentBinaryImagesStart + AMDesc.NumJITBinaries;
    CurrentBinaryImagesStart +=
        AMDesc.NumJITBinaries + AM.NativeDeviceCodeImages.size();

    auto &BinPropertySets = convertAbstractModuleProperties(AM);

    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const auto &IRM = AM.IRModules[J];

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_SPIRV; // TODO: Determine.
      DeviceBinary.DeviceTargetSpec =
          __SYCL_DEVICE_BINARY_TARGET_SPIRV64; // TODO: Determine.
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.BinaryStart =
          reinterpret_cast<const unsigned char *>(IRM.RawIRBytes.data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          IRM.RawIRBytes.data() + IRM.RawIRBytes.size());
      DeviceBinary.EntriesBegin = nullptr;
      DeviceBinary.EntriesEnd = nullptr;
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      AMDesc.JITBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }

    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const auto &NDCI = AM.NativeDeviceCodeImages[J];

      assert(NDCI.Metadata != nullptr);
      auto &NDCIMetadataProps =
          (*NDCI.Metadata)[llvm::util::PropertySetRegistry::
                               SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA];

      auto &TargetTripleProp = NDCIMetadataProps[llvm::StringRef("target")];
      std::string_view TargetTriple{
          reinterpret_cast<const char *>(TargetTripleProp.asByteArray()),
          static_cast<size_t>(TargetTripleProp.getByteArraySize())};

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_NATIVE;
      DeviceBinary.DeviceTargetSpec =
          getDeviceTargetSpecFromTriple(TargetTriple);
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.BinaryStart = reinterpret_cast<const unsigned char *>(
          NDCI.RawDeviceCodeImageBytes.data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          NDCI.RawDeviceCodeImageBytes.data() +
          NDCI.RawDeviceCodeImageBytes.size());
      DeviceBinary.EntriesBegin = nullptr;
      DeviceBinary.EntriesEnd = nullptr;
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      AMDesc.NativeBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }
  }
}

std::vector<_sycl_device_binary_property_set_struct> &
SYCLBINBinaries::Impl::convertAbstractModuleProperties(
    const llvm::object::SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_device_binary_property_set_struct> &BinPropSets =
      BinaryPropertySets.emplace_back();
  BinPropSets.reserve(AM.Metadata->getPropSets().size());
  for (auto &PropSetIt : *AM.Metadata) {
    auto &PropSetName = PropSetIt.first;
    auto &PropSetVal = PropSetIt.second;

    std::vector<_sycl_device_binary_property_struct> &PropsList =
        BinaryProperties.emplace_back();
    PropsList.reserve(PropSetVal.size());

    for (auto &PropIt : PropSetVal) {
      auto &PropName = PropIt.first;
      auto &PropVal = PropIt.second;
      _sycl_device_binary_property_struct &BinProp = PropsList.emplace_back();
      BinProp.Name = const_cast<char *>(PropName.data());
      BinProp.Type = static_cast<uint32_t>(PropVal.getType());
      if (BinProp.Type == SYCL_PROPERTY_TYPE_UINT32) {
        // UINT32 properties have their value stored in the size instead.
        BinProp.ValAddr = nullptr;
        std::memcpy(&BinProp.ValSize, PropVal.data(), sizeof(uint32_t));
      } else {
        BinProp.ValAddr = const_cast<char *>(PropVal.data());
        BinProp.ValSize = PropVal.size();
      }
    }

    _sycl_device_binary_property_set_struct &BinPropSet =
        BinPropSets.emplace_back();
    BinPropSet.Name = const_cast<char *>(PropSetName.data());
    BinPropSet.PropertiesBegin = PropsList.data();
    BinPropSet.PropertiesEnd = PropsList.data() + PropsList.size();
  }
  return BinPropSets;
}

SYCLBINBinaries::SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize)
    : PImpl{std::make_unique<Impl>(SYCLBINContent, SYCLBINSize)} {}

SYCLBINBinaries::SYCLBINBinaries(SYCLBINBinaries &&) noexcept = default;
SYCLBINBinaries &
SYCLBINBinaries::operator=(SYCLBINBinaries &&) noexcept = default;
SYCLBINBinaries::~SYCLBINBinaries() = default;

uint8_t SYCLBINBinaries::getState() const { return PImpl->getState(); }

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(device_impl &Dev, bundle_state State) {
  auto GetCompatibleImage = [&](const RTDeviceBinaryImage *Imgs,
                                size_t NumImgs) {
    const RTDeviceBinaryImage *CompatImagePtr =
        std::find_if(Imgs, Imgs + NumImgs, [&](const RTDeviceBinaryImage &Img) {
          return doesDevSupportDeviceRequirements(Dev, Img) &&
                 doesImageTargetMatchDevice(Img, Dev);
        });
    return (CompatImagePtr != Imgs + NumImgs) ? CompatImagePtr : nullptr;
  };

  std::vector<const RTDeviceBinaryImage *> Images;
  for (size_t I = 0; I < PImpl->getNumAbstractModules(); ++I) {
    const auto &AMDesc = PImpl->AbstractModuleDescriptors[I];
    if (State == bundle_state::executable) {
      if (const RTDeviceBinaryImage *CompatImagePtr = GetCompatibleImage(
              AMDesc.NativeBinaries, AMDesc.NumNativeBinaries)) {
        Images.push_back(CompatImagePtr);
        continue;
      }
    }

    if (const RTDeviceBinaryImage *CompatImagePtr =
            GetCompatibleImage(AMDesc.JITBinaries, AMDesc.NumJITBinaries))
      Images.push_back(CompatImagePtr);
  }
  return Images;
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(devices_range Devs,
                                         bundle_state State) {
  std::set<const RTDeviceBinaryImage *> Images;
  for (device_impl &Dev : Devs) {
    std::vector<const RTDeviceBinaryImage *> BestImagesForDev =
        getBestCompatibleImages(Dev, State);
    Images.insert(BestImagesForDev.cbegin(), BestImagesForDev.cend());
  }
  return {Images.cbegin(), Images.cend()};
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getNativeBinaryImages(device_impl &Dev) {
  std::vector<const RTDeviceBinaryImage *> Images;
  for (size_t I = 0; I < PImpl->getNumAbstractModules(); ++I) {
    const auto &AMDesc = PImpl->AbstractModuleDescriptors[I];

    const RTDeviceBinaryImage *CompatImagePtr = std::find_if(
        AMDesc.NativeBinaries, AMDesc.NativeBinaries + AMDesc.NumNativeBinaries,
        [&](const RTDeviceBinaryImage &Img) {
          return doesDevSupportDeviceRequirements(Dev, Img) &&
                 doesImageTargetMatchDevice(Img, Dev);
        });
    if (CompatImagePtr != AMDesc.NativeBinaries + AMDesc.NumNativeBinaries)
      Images.push_back(CompatImagePtr);
  }
  return Images;
}

#else // SYCL_RT_HAS_LLVMOBJECT

// Stub implementation for runtime flavours that cannot link LLVMObject (for
// example, the MSVC sycl runtime DLL whose CRT does not match the CRT used to
// build the LLVM libraries). Any attempt to construct or query a
// SYCLBINBinaries throws errc::feature_not_supported -- the runtime can still
// link and operate, just without SYCLBIN parsing support.

struct SYCLBINBinaries::Impl {};

[[noreturn]] static void throwUnsupported() {
  throw sycl::exception(
      make_error_code(errc::feature_not_supported),
      "SYCLBIN parsing is not available in this build of the SYCL runtime "
      "library (LLVM SYCLBIN reader was not linked into this runtime "
      "flavour).");
}

SYCLBINBinaries::SYCLBINBinaries(const char *, size_t) { throwUnsupported(); }

SYCLBINBinaries::SYCLBINBinaries(SYCLBINBinaries &&) noexcept = default;
SYCLBINBinaries &
SYCLBINBinaries::operator=(SYCLBINBinaries &&) noexcept = default;
SYCLBINBinaries::~SYCLBINBinaries() = default;

uint8_t SYCLBINBinaries::getState() const { throwUnsupported(); }

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(device_impl &, bundle_state) {
  throwUnsupported();
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getBestCompatibleImages(devices_range, bundle_state) {
  throwUnsupported();
}

std::vector<const RTDeviceBinaryImage *>
SYCLBINBinaries::getNativeBinaryImages(device_impl &) { throwUnsupported(); }

#endif // SYCL_RT_HAS_LLVMOBJECT

} // namespace detail
} // namespace _V1
} // namespace sycl
