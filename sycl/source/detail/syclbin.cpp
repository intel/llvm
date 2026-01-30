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

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace {

std::unique_ptr<char[]> ContentCopy(const char *Data, size_t Size) {
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

} // namespace

SYCLBINBinaries::SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize)
    : SYCLBINContentCopy{ContentCopy(SYCLBINContent, SYCLBINSize)} {
  llvm::MemoryBufferRef ContentRef(
      llvm::StringRef(SYCLBINContentCopy.get(), SYCLBINSize), "");

  // Try to parse as SYCLBIN first
  auto NewSYCLBINOrErr = llvm::object::SYCLBIN::read(ContentRef);
  if (NewSYCLBINOrErr) {
    ParsedSYCLBIN = std::move(*NewSYCLBINOrErr);
  } else {
    // Try legacy format for backward compatibility.
    auto OffloadBinaryVecOrError =
        llvm::object::OffloadBinary::create(ContentRef);
    if (!OffloadBinaryVecOrError) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Failed to parse SYCLBIN file: invalid format");
    }

    auto OldSYCLBINOrErr = llvm::object::SYCLBIN::read(llvm::MemoryBufferRef(
        OffloadBinaryVecOrError->front()->getImage(), ""));
    if (!OldSYCLBINOrErr) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Failed to parse SYCLBIN file: invalid format");
    }
    ParsedSYCLBIN = std::move(*OldSYCLBINOrErr);
  }

  AbstractModuleDescriptors = std::unique_ptr<AbstractModuleDesc[]>(
      new AbstractModuleDesc[ParsedSYCLBIN->AbstractModules.size()]);

  size_t NumBinaries = 0;
  for (const llvm::object::SYCLBIN::AbstractModule &AM :
       ParsedSYCLBIN->AbstractModules)
    NumBinaries += AM.IRModules.size() + AM.NativeDeviceCodeImages.size();
  DeviceBinaries.reserve(NumBinaries);
  BinaryImages = std::unique_ptr<RTDeviceBinaryImage[]>(
      new RTDeviceBinaryImage[NumBinaries]);

  RTDeviceBinaryImage *CurrentBinaryImagesStart = BinaryImages.get();
  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    llvm::object::SYCLBIN::AbstractModule &AM =
        ParsedSYCLBIN->AbstractModules[I];
    AbstractModuleDesc &AMDesc = AbstractModuleDescriptors[I];

    // Set up the abstract module descriptor.
    AMDesc.NumJITBinaries = AM.IRModules.size();
    AMDesc.NumNativeBinaries = AM.NativeDeviceCodeImages.size();
    AMDesc.JITBinaries = CurrentBinaryImagesStart;
    AMDesc.NativeBinaries = CurrentBinaryImagesStart + AMDesc.NumJITBinaries;
    CurrentBinaryImagesStart +=
        AMDesc.NumJITBinaries + AM.NativeDeviceCodeImages.size();

    // Construct properties from SYCLBIN metadata.
    std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
        convertAbstractModuleProperties(AM);

    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const llvm::object::OffloadBinary *IRM = AM.IRModules[J];

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_SPIRV; // TODO: Determine.
      DeviceBinary.DeviceTargetSpec =
          __SYCL_DEVICE_BINARY_TARGET_SPIRV64; // TODO: Determine.
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.BinaryStart =
          reinterpret_cast<const unsigned char *>(IRM->getImage().data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          IRM->getImage().data() + IRM->getImage().size());
      DeviceBinary.EntriesBegin = nullptr;
      DeviceBinary.EntriesEnd = nullptr;
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      // Create an image from it.
      AMDesc.JITBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }

    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const llvm::object::OffloadBinary *NDCI = AM.NativeDeviceCodeImages[J];

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_NATIVE;
      DeviceBinary.DeviceTargetSpec =
          getDeviceTargetSpecFromTriple(NDCI->getTriple());
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
      DeviceBinary.BinaryStart =
          reinterpret_cast<const unsigned char *>(NDCI->getImage().data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          NDCI->getImage().data() + NDCI->getImage().size());
      DeviceBinary.EntriesBegin = nullptr;
      DeviceBinary.EntriesEnd = nullptr;
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      // Create an image from it.
      AMDesc.NativeBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }
  }
}

std::vector<_sycl_device_binary_property_set_struct> &
SYCLBINBinaries::convertAbstractModuleProperties(
    llvm::object::SYCLBIN::AbstractModule &AM) {
  std::vector<_sycl_device_binary_property_set_struct> &BinPropertySets =
      BinaryPropertySets.emplace_back();
  BinPropertySets.reserve(AM.Metadata->getPropSets().size());
  for (auto &PropSetIt : *AM.Metadata) {
    auto &PropSetName = PropSetIt.first;
    auto &PropSetVal = PropSetIt.second;

    // Add a new vector to BinaryProperties and reserve room for all the
    // properties we are converting.
    std::vector<_sycl_device_binary_property_struct> &PropsList =
        BinaryProperties.emplace_back();
    PropsList.reserve(PropSetVal.size());

    // Then convert all properties in the property set.
    for (auto &PropIt : PropSetVal) {
      auto &PropName = PropIt.first;
      auto &PropVal = PropIt.second;
      _sycl_device_binary_property_struct &BinProp = PropsList.emplace_back();
      BinProp.Name = const_cast<char *>(PropName.data());
      BinProp.Type = PropVal.getType();
      if (BinProp.Type == SYCL_PROPERTY_TYPE_UINT32) {
        // UINT32 properties have their value stored in the size instead.
        BinProp.ValAddr = nullptr;
        std::memcpy(&BinProp.ValSize, PropVal.data(), sizeof(uint32_t));
      } else {
        BinProp.ValAddr = const_cast<char *>(PropVal.data());
        BinProp.ValSize = PropVal.size();
      }
    }

    // Add a new property set to the list.
    _sycl_device_binary_property_set_struct &BinPropSet =
        BinPropertySets.emplace_back();
    BinPropSet.Name = const_cast<char *>(PropSetName.data());
    BinPropSet.PropertiesBegin = PropsList.data();
    BinPropSet.PropertiesEnd = PropsList.data() + PropsList.size();
  }
  return BinPropertySets;
}

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
  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    const AbstractModuleDesc &AMDesc = AbstractModuleDescriptors[I];
    // If the target state is executable, try with native images first.
    if (State == bundle_state::executable) {
      if (const RTDeviceBinaryImage *CompatImagePtr = GetCompatibleImage(
              AMDesc.NativeBinaries, AMDesc.NumNativeBinaries)) {
        Images.push_back(CompatImagePtr);
        continue;
      }
    }

    // Otherwise, select the first compatible JIT binary.
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
  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    const AbstractModuleDesc &AMDesc = AbstractModuleDescriptors[I];
    // If the target state is executable, try with native images first.

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

} // namespace detail
} // namespace _V1
} // namespace sycl
