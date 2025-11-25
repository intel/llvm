//==--------------------- syclbin.cpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Adjusted copy of llvm/lib/Object/SYCLBIN.cpp.
// TODO: Remove once we can consistently link the SYCL runtime library with
// LLVMObject.

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

// Offload binary header and entry.
constexpr uint8_t OffloadBinaryMagic[4] = {0x10, 0xFF, 0x10, 0xAD};
struct OffloadBinaryHeaderType {
  uint8_t Magic[4];
  uint32_t Version;
  uint64_t Size;
  uint64_t EntryOffset;
  uint64_t EntrySize;
};
struct OffloadBinaryEntryType {
  uint16_t ImageKind;
  uint16_t OffloadKind;
  uint32_t Flags;
  uint64_t StringOffset;
  uint64_t NumStrings;
  uint64_t ImageOffset;
  uint64_t ImageSize;
};

class BlockReader {
protected:
  BlockReader(const char *Data, size_t Size) : Data{Data}, Size{Size} {}

  void ReadSizeCheck(size_t ByteOffset, size_t ReadSize) {
    if (ByteOffset + ReadSize > Size)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Unexpected file contents size.");
  }

  const char *Data = nullptr;
  size_t Size = 0;
};

class HeaderBlockReader : public BlockReader {
public:
  HeaderBlockReader(const char *Data, size_t Size) : BlockReader(Data, Size) {}

  template <typename HeaderT> const HeaderT *GetHeaderPtr(size_t ByteOffset) {
    ReadSizeCheck(ByteOffset, sizeof(HeaderT));
    return reinterpret_cast<const HeaderT *>(Data + ByteOffset);
  }
};

class SYCLBINByteTableBlockReader : public BlockReader {
public:
  SYCLBINByteTableBlockReader(const char *Data, size_t Size)
      : BlockReader(Data, Size) {}

  std::string_view GetBinaryBlob(size_t ByteOffset, uint64_t BlobSize) {
    ReadSizeCheck(ByteOffset, BlobSize);
    return {Data + ByteOffset, BlobSize};
  }

  std::unique_ptr<PropertySetRegistry> GetMetadata(size_t ByteOffset,
                                                   uint64_t MetadataSize) {
    return PropertySetRegistry::read(GetBinaryBlob(ByteOffset, MetadataSize));
  }
};

std::pair<const char *, size_t> getImageInOffloadBinary(const char *Data,
                                                        size_t Size) {
  if (sizeof(OffloadBinaryHeaderType) > Size)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Invalid Offload Binary size.");

  // Read the header.
  const OffloadBinaryHeaderType *Header =
      reinterpret_cast<const OffloadBinaryHeaderType *>(Data);
  if (memcmp(Header->Magic, OffloadBinaryMagic, 4) != 0)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Incorrect Offload Binary magic number.");

  if (Header->Version != 1)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unsupported Offload Binary version number.");

  if (Header->EntrySize != sizeof(OffloadBinaryEntryType))
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unexpected number of offload entries.");

  if (Header->EntryOffset + sizeof(OffloadBinaryEntryType) > Size)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Invalid entry offset.");

  // Read the table entry.
  const OffloadBinaryEntryType *Entry =
      reinterpret_cast<const OffloadBinaryEntryType *>(Data +
                                                       Header->EntryOffset);

  if (Entry->ImageKind != /*IMG_SYCLBIN*/ 7)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unexpected image type.");

  if (Entry->ImageOffset + Entry->ImageSize > Size)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Invalid image offset and size.");

  return std::make_pair(Data + Entry->ImageOffset, Entry->ImageSize);
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
  if (Target == __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA)
    return __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NVPTX64)
    return __SYCL_DEVICE_BINARY_TARGET_NVPTX64;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_AMDGCN)
    return __SYCL_DEVICE_BINARY_TARGET_AMDGCN;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU)
    return __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU;
  return UR_DEVICE_BINARY_TARGET_UNKNOWN;
}

} // namespace

SYCLBIN::SYCLBIN(const char *Data, size_t Size) {
  auto [SYCLBINData, SYCLBINSize] = getImageInOffloadBinary(Data, Size);

  if (SYCLBINSize < sizeof(FileHeaderType))
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unexpected file contents size.");

  // Read the file header.
  const FileHeaderType *FileHeader =
      reinterpret_cast<const FileHeaderType *>(SYCLBINData);
  if (FileHeader->Magic != MagicNumber)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Incorrect SYCLBIN magic number.");

  if (FileHeader->Version > CurrentVersion)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unsupported SYCLBIN version " +
                              std::to_string(FileHeader->Version) + ".");
  Version = FileHeader->Version;

  const uint64_t AMHeaderBlockSize =
      sizeof(AbstractModuleHeaderType) * FileHeader->AbstractModuleCount;
  const uint64_t IRMHeaderBlockSize =
      sizeof(IRModuleHeaderType) * FileHeader->IRModuleCount;
  const uint64_t NDCIHeaderBlockSize = sizeof(NativeDeviceCodeImageHeaderType) *
                                       FileHeader->NativeDeviceCodeImageCount;
  const uint64_t HeaderBlockSize = sizeof(FileHeaderType) + AMHeaderBlockSize +
                                   IRMHeaderBlockSize + NDCIHeaderBlockSize;
  // Align metadata table size to 8.
  const uint64_t AlignedMetadataByteTableSize =
      FileHeader->MetadataByteTableSize +
      (-FileHeader->MetadataByteTableSize & 7);
  if (SYCLBINSize < HeaderBlockSize + AlignedMetadataByteTableSize +
                        FileHeader->BinaryByteTableSize)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unexpected file contents size.");

  // Create reader objects. These help with checking out-of-bounds access.
  HeaderBlockReader HeaderBlockReader{SYCLBINData, HeaderBlockSize};
  SYCLBINByteTableBlockReader MetadataByteTableBlockReader{
      SYCLBINData + HeaderBlockSize, FileHeader->MetadataByteTableSize};
  SYCLBINByteTableBlockReader BinaryByteTableBlockReader{
      SYCLBINData + HeaderBlockSize + AlignedMetadataByteTableSize,
      FileHeader->BinaryByteTableSize};

  // Read global metadata.
  GlobalMetadata = MetadataByteTableBlockReader.GetMetadata(
      FileHeader->GlobalMetadataOffset, FileHeader->GlobalMetadataSize);

  // Read the abstract modules.
  AbstractModules.resize(FileHeader->AbstractModuleCount);
  for (uint32_t I = 0; I < FileHeader->AbstractModuleCount; ++I) {
    AbstractModule &AM = AbstractModules[I];

    // Read the header for the current abstract module.
    const uint64_t AMHeaderByteOffset =
        sizeof(FileHeaderType) + sizeof(AbstractModuleHeaderType) * I;
    const AbstractModuleHeaderType *AMHeader =
        HeaderBlockReader.GetHeaderPtr<AbstractModuleHeaderType>(
            AMHeaderByteOffset);

    // Read the metadata for the current abstract module.
    AM.Metadata = MetadataByteTableBlockReader.GetMetadata(
        AMHeader->MetadataOffset, AMHeader->MetadataSize);

    // Read the IR modules of the current abstract module.
    AM.IRModules.resize(AMHeader->IRModuleCount);
    for (uint32_t J = 0; J < AMHeader->IRModuleCount; ++J) {
      IRModule &IRM = AM.IRModules[J];

      // Read the header for the current IR module.
      const uint64_t IRMHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize +
          sizeof(IRModuleHeaderType) * (AMHeader->IRModuleOffset + J);
      const IRModuleHeaderType *IRMHeader =
          HeaderBlockReader.GetHeaderPtr<IRModuleHeaderType>(
              IRMHeaderByteOffset);

      // Read the metadata for the current IR module.
      IRM.Metadata = MetadataByteTableBlockReader.GetMetadata(
          IRMHeader->MetadataOffset, IRMHeader->MetadataSize);

      // Read the binary blob for the current IR module.
      IRM.RawIRBytes = BinaryByteTableBlockReader.GetBinaryBlob(
          IRMHeader->RawIRBytesOffset, IRMHeader->RawIRBytesSize);
    }

    // Read the native device code images of the current abstract module.
    AM.NativeDeviceCodeImages.resize(AMHeader->NativeDeviceCodeImageCount);
    for (uint32_t J = 0; J < AMHeader->NativeDeviceCodeImageCount; ++J) {
      NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[J];

      // Read the header for the current native device code image.
      const uint64_t NDCIHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize + IRMHeaderBlockSize +
          sizeof(NativeDeviceCodeImageHeaderType) *
              (AMHeader->NativeDeviceCodeImageOffset + J);
      const NativeDeviceCodeImageHeaderType *NDCIHeader =
          HeaderBlockReader.GetHeaderPtr<NativeDeviceCodeImageHeaderType>(
              NDCIHeaderByteOffset);

      // Read the metadata for the current native device code image.
      NDCI.Metadata = MetadataByteTableBlockReader.GetMetadata(
          NDCIHeader->MetadataOffset, NDCIHeader->MetadataSize);

      // Read the binary blob for the current native device code image.
      NDCI.RawDeviceCodeImageBytes = BinaryByteTableBlockReader.GetBinaryBlob(
          NDCIHeader->BinaryBytesOffset, NDCIHeader->BinaryBytesSize);
    }
  }
}

SYCLBINBinaries::SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize)
    : SYCLBINContentCopy{ContentCopy(SYCLBINContent, SYCLBINSize)},
      SYCLBINContentCopySize{SYCLBINSize},
      ParsedSYCLBIN(SYCLBIN{SYCLBINContentCopy.get(), SYCLBINSize}) {
  AbstractModuleDescriptors = std::unique_ptr<AbstractModuleDesc[]>(
      new AbstractModuleDesc[ParsedSYCLBIN.AbstractModules.size()]);

  size_t NumBinaries = 0;
  for (const SYCLBIN::AbstractModule &AM : ParsedSYCLBIN.AbstractModules)
    NumBinaries += AM.IRModules.size() + AM.NativeDeviceCodeImages.size();
  DeviceBinaries.reserve(NumBinaries);
  BinaryImages = std::unique_ptr<RTDeviceBinaryImage[]>(
      new RTDeviceBinaryImage[NumBinaries]);

  RTDeviceBinaryImage *CurrentBinaryImagesStart = BinaryImages.get();
  for (size_t I = 0; I < getNumAbstractModules(); ++I) {
    SYCLBIN::AbstractModule &AM = ParsedSYCLBIN.AbstractModules[I];
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
      SYCLBIN::IRModule &IRM = AM.IRModules[J];

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_SPIRV; // TODO: Determine.
      DeviceBinary.DeviceTargetSpec =
          __SYCL_DEVICE_BINARY_TARGET_SPIRV64; // TODO: Determine.
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
      DeviceBinary.ManifestStart = nullptr;
      DeviceBinary.ManifestEnd = nullptr;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
      DeviceBinary.BinaryStart =
          reinterpret_cast<const unsigned char *>(IRM.RawIRBytes.data());
      DeviceBinary.BinaryEnd = reinterpret_cast<const unsigned char *>(
          IRM.RawIRBytes.data() + IRM.RawIRBytes.size());
      DeviceBinary.EntriesBegin = nullptr;
      DeviceBinary.EntriesEnd = nullptr;
      DeviceBinary.PropertySetsBegin = BinPropertySets.data();
      DeviceBinary.PropertySetsEnd =
          BinPropertySets.data() + BinPropertySets.size();
      // Create an image from it.
      AMDesc.JITBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }

    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const SYCLBIN::NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[J];

      assert(NDCI.Metadata != nullptr);
      PropertySet &NDCIMetadataProps = (*NDCI.Metadata)
          [PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA];

      auto &TargetTripleProp = NDCIMetadataProps["target"];
      std::string_view TargetTriple = std::string_view{
          reinterpret_cast<const char *>(TargetTripleProp.asByteArray()),
          TargetTripleProp.getByteArraySize()};

      sycl_device_binary_struct &DeviceBinary = DeviceBinaries.emplace_back();
      DeviceBinary.Version = SYCL_DEVICE_BINARY_VERSION;
      DeviceBinary.Kind = 4;
      DeviceBinary.Format = SYCL_DEVICE_BINARY_TYPE_NATIVE;
      DeviceBinary.DeviceTargetSpec =
          getDeviceTargetSpecFromTriple(TargetTriple);
      DeviceBinary.CompileOptions = nullptr;
      DeviceBinary.LinkOptions = nullptr;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
      DeviceBinary.ManifestStart = nullptr;
      DeviceBinary.ManifestEnd = nullptr;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
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
      // Create an image from it.
      AMDesc.NativeBinaries[J] = RTDeviceBinaryImage{&DeviceBinary};
    }
  }
}

std::vector<_sycl_device_binary_property_set_struct> &
SYCLBINBinaries::convertAbstractModuleProperties(SYCLBIN::AbstractModule &AM) {
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
  for (size_t I = 0; I < getNumAbstractModules(); ++I) {
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
  for (size_t I = 0; I < getNumAbstractModules(); ++I) {
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
