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

#include <sstream>

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
  uint64_t EntriesOffset; // V2: Renamed from EntryOffset
  uint64_t EntriesCount;  // V2: Renamed from EntrySize, now stores count
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

  // Support both v1 and v2 formats
  if (Header->Version == 0 || Header->Version > 2)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unsupported Offload Binary version number.");

  // V1: EntriesCount was EntrySize and stored sizeof(Entry)
  // V2: EntriesCount stores the number of entries
  uint64_t EntriesCount = (Header->Version == 1) ? 1 : Header->EntriesCount;
  uint64_t EntriesSize = sizeof(OffloadBinaryEntryType) * EntriesCount;

  if (Header->Version == 1 &&
      Header->EntriesCount != sizeof(OffloadBinaryEntryType))
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unexpected entry size for v1 format.");

  if (Header->EntriesOffset + EntriesSize > Size)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Invalid entries offset.");

  // Read the first entry (for SYCLBIN, we expect a single entry)
  const OffloadBinaryEntryType *Entry =
      reinterpret_cast<const OffloadBinaryEntryType *>(Data +
                                                       Header->EntriesOffset);

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
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NVPTX64)
    return __SYCL_DEVICE_BINARY_TARGET_NVPTX64;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_AMDGCN)
    return __SYCL_DEVICE_BINARY_TARGET_AMDGCN;
  if (Target == __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU)
    return __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU;
  return UR_DEVICE_BINARY_TARGET_UNKNOWN;
}

// Returns the triple string corresponding to the given runtime
// DeviceTargetSpec. The SYCLBIN reader only inspects the first '-'-separated
// segment of the triple to recover the target spec, so emitting
// "<spec>-unknown-unknown" is sufficient for round-tripping.
std::string getTripleFromTargetSpec(const char *TargetSpec) {
  return std::string{TargetSpec ? TargetSpec : ""} + "-unknown-unknown";
}

// Determines whether the given device image format should be packaged as an IR
// module or a native device code image inside a SYCLBIN.
bool isIRModuleFormat(ur::DeviceBinaryType Format) {
  return Format == SYCL_DEVICE_BINARY_TYPE_SPIRV ||
         Format == SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE;
}

// Maps a runtime device image format to its SYCLBIN IR module type enum
// value. The list of valid IR types is currently SPIR-V (0), PTX (1) and
// AMDGCN (2). Today the runtime only emits SPIR-V IR modules, so we hardcode 0
// here. Mirrors the same hardcode in llvm/lib/Object/SYCLBIN.cpp.
// TODO: Determine type from the input.
uint32_t getIRModuleType(ur::DeviceBinaryType /*Format*/) { return 0; }

// Serializes a property set (name + entries) into the textual format consumed
// by PropertySetRegistry::read by routing through PropertySetRegistry::write.
// The source data lives in a sycl_device_binary_property_set_struct produced by
// either offline compilation or the SYCLBIN reader.
void appendPropSetTo(PropertySetRegistry &Registry,
                     sycl_device_binary_property_set PropSet) {
  std::string_view Category{PropSet->Name};
  PropertySet &Set = Registry[Category];
  for (sycl_device_binary_property Prop = PropSet->PropertiesBegin;
       Prop != PropSet->PropertiesEnd; ++Prop) {
    PropertyValue::Type Ty =
        PropertyValue::getTypeTag(static_cast<int>(Prop->Type));
    PropertyValue Value(Ty);
    if (Ty == PropertyValue::UINT32) {
      // For UINT32 properties the value is encoded directly in ValSize.
      Value.set(static_cast<uint32_t>(Prop->ValSize));
    } else {
      // BYTE_ARRAY: Prop->ValAddr is laid out as [8-byte size header][payload]
      // and Prop->ValSize covers the entire blob. Recompute payload size from
      // ValSize rather than reading the header (the header is the encoded
      // bit-count and is not always set consistently in synthetic images).
      assert(Prop->ValSize >= sizeof(PropertyValue::SizeTy) &&
             "BYTE_ARRAY property smaller than its 8-byte size header.");
      const PropertyValue::byte *Payload =
          reinterpret_cast<const PropertyValue::byte *>(Prop->ValAddr) +
          sizeof(PropertyValue::SizeTy);
      const PropertyValue::SizeTy PayloadBytes =
          Prop->ValSize - sizeof(PropertyValue::SizeTy);
      Value = PropertyValue(Payload, PayloadBytes * 8);
    }
    Set[std::string{Prop->Name}] = std::move(Value);
  }
}

// Serializes a PropertySetRegistry to a std::string suitable for stuffing into
// a SYCLBIN metadata blob.
std::string serializePropSetRegistry(const PropertySetRegistry &Registry) {
  std::ostringstream OS;
  Registry.write(OS);
  return OS.str();
}

// Pads OS so that its current size is a multiple of 8 bytes by appending zero
// bytes.
void padTo8(std::ostream &OS) {
  size_t Pos = static_cast<size_t>(OS.tellp());
  size_t Pad = (8 - (Pos & 7)) & 7;
  for (size_t I = 0; I < Pad; ++I)
    OS.put('\0');
}

template <typename T> void writeRaw(std::ostream &OS, const T &Val) {
  OS.write(reinterpret_cast<const char *>(&Val), sizeof(T));
}

} // namespace

std::vector<char> SYCLBIN::write(const SYCLBINDesc &Desc) {
  uint32_t IRModuleCount = 0;
  uint32_t NativeDeviceCodeImageCount = 0;
  uint64_t MetadataTableSize = Desc.GlobalMetadata.size();
  uint64_t BinaryTableSize = 0;
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules) {
    IRModuleCount += AMD.IRModules.size();
    NativeDeviceCodeImageCount += AMD.NativeDeviceCodeImages.size();
    MetadataTableSize += AMD.Metadata.size();
    for (const ImageDesc &IRMD : AMD.IRModules) {
      MetadataTableSize += IRMD.Metadata.size();
      BinaryTableSize += IRMD.Bytes.size();
    }
    for (const ImageDesc &NDCID : AMD.NativeDeviceCodeImages) {
      MetadataTableSize += NDCID.Metadata.size();
      BinaryTableSize += NDCID.Bytes.size();
    }
  }

  std::ostringstream OS;

  // File header.
  FileHeaderType FileHeader{};
  FileHeader.Magic = MagicNumber;
  FileHeader.Version = CurrentVersion;
  FileHeader.AbstractModuleCount =
      static_cast<uint32_t>(Desc.AbstractModules.size());
  FileHeader.IRModuleCount = IRModuleCount;
  FileHeader.NativeDeviceCodeImageCount = NativeDeviceCodeImageCount;
  FileHeader.MetadataByteTableSize = MetadataTableSize;
  FileHeader.BinaryByteTableSize = BinaryTableSize;
  FileHeader.GlobalMetadataOffset = 0;
  FileHeader.GlobalMetadataSize = Desc.GlobalMetadata.size();
  writeRaw(OS, FileHeader);
  padTo8(OS);

  // Track running offsets into the metadata and binary byte tables.
  uint64_t MetadataOffset = FileHeader.GlobalMetadataSize;
  uint64_t BinaryOffset = 0;

  // Abstract module headers.
  uint32_t IRModuleOffset = 0;
  uint32_t NativeDeviceCodeImageOffset = 0;
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules) {
    AbstractModuleHeaderType AMHeader{};
    AMHeader.MetadataOffset = MetadataOffset;
    AMHeader.MetadataSize = AMD.Metadata.size();
    AMHeader.IRModuleCount = static_cast<uint32_t>(AMD.IRModules.size());
    AMHeader.IRModuleOffset = IRModuleOffset;
    AMHeader.NativeDeviceCodeImageCount =
        static_cast<uint32_t>(AMD.NativeDeviceCodeImages.size());
    AMHeader.NativeDeviceCodeImageOffset = NativeDeviceCodeImageOffset;
    writeRaw(OS, AMHeader);
    padTo8(OS);
    MetadataOffset += AMHeader.MetadataSize;
    IRModuleOffset += AMHeader.IRModuleCount;
    NativeDeviceCodeImageOffset += AMHeader.NativeDeviceCodeImageCount;
  }

  // IR module headers.
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules) {
    for (const ImageDesc &IRMD : AMD.IRModules) {
      IRModuleHeaderType IRMHeader{};
      IRMHeader.MetadataOffset = MetadataOffset;
      IRMHeader.MetadataSize = IRMD.Metadata.size();
      IRMHeader.RawIRBytesOffset = BinaryOffset;
      IRMHeader.RawIRBytesSize = IRMD.Bytes.size();
      writeRaw(OS, IRMHeader);
      padTo8(OS);
      MetadataOffset += IRMHeader.MetadataSize;
      BinaryOffset += IRMHeader.RawIRBytesSize;
    }
  }

  // Native device code image headers.
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules) {
    for (const ImageDesc &NDCID : AMD.NativeDeviceCodeImages) {
      NativeDeviceCodeImageHeaderType NDCIHeader{};
      NDCIHeader.MetadataOffset = MetadataOffset;
      NDCIHeader.MetadataSize = NDCID.Metadata.size();
      NDCIHeader.BinaryBytesOffset = BinaryOffset;
      NDCIHeader.BinaryBytesSize = NDCID.Bytes.size();
      writeRaw(OS, NDCIHeader);
      padTo8(OS);
      MetadataOffset += NDCIHeader.MetadataSize;
      BinaryOffset += NDCIHeader.BinaryBytesSize;
    }
  }

  // Metadata byte table.
  OS.write(Desc.GlobalMetadata.data(), Desc.GlobalMetadata.size());
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules)
    OS.write(AMD.Metadata.data(), AMD.Metadata.size());
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules)
    for (const ImageDesc &IRMD : AMD.IRModules)
      OS.write(IRMD.Metadata.data(), IRMD.Metadata.size());
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules)
    for (const ImageDesc &NDCID : AMD.NativeDeviceCodeImages)
      OS.write(NDCID.Metadata.data(), NDCID.Metadata.size());
  padTo8(OS);

  // Binary byte table. Order must match the offsets baked into the IR and
  // native module headers above: all IR module bytes first, then all native
  // device code image bytes.
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules)
    for (const ImageDesc &IRMD : AMD.IRModules)
      OS.write(IRMD.Bytes.data(), IRMD.Bytes.size());
  for (const AbstractModuleDesc &AMD : Desc.AbstractModules)
    for (const ImageDesc &NDCID : AMD.NativeDeviceCodeImages)
      OS.write(NDCID.Bytes.data(), NDCID.Bytes.size());
  padTo8(OS);

  // The runtime SYCLBIN reader expects the raw SYCLBIN bytes to be wrapped in
  // an OffloadBinary entry, so emit that wrapping here. Mirrors the wrapping
  // that clang-linker-wrapper does for on-disk SYCLBIN files.
  std::string SYCLBINBytes = OS.str();
  std::ostringstream OB{std::ios::binary};
  OffloadBinaryHeaderType OBHeader{};
  std::memcpy(OBHeader.Magic, OffloadBinaryMagic, sizeof(OBHeader.Magic));
  OBHeader.Version = 2;
  OBHeader.EntriesOffset = sizeof(OffloadBinaryHeaderType);
  OBHeader.EntriesCount = 1;
  // Single entry pointing at the SYCLBIN payload following the entry array.
  OffloadBinaryEntryType OBEntry{};
  OBEntry.ImageKind = /*IMG_SYCLBIN*/ 7;
  OBEntry.OffloadKind = /*OFK_SYCL*/ 5;
  OBEntry.ImageOffset =
      sizeof(OffloadBinaryHeaderType) + sizeof(OffloadBinaryEntryType);
  OBEntry.ImageSize = SYCLBINBytes.size();
  OBHeader.Size = OBEntry.ImageOffset + SYCLBINBytes.size();
  writeRaw(OB, OBHeader);
  writeRaw(OB, OBEntry);
  OB.write(SYCLBINBytes.data(), SYCLBINBytes.size());

  std::string Buf = OB.str();
  return std::vector<char>(Buf.begin(), Buf.end());
}

std::vector<char>
SYCLBIN::serializeImages(const std::vector<const RTDeviceBinaryImage *> &Images,
                         uint8_t State) {
  SYCLBINDesc Desc;

  // Global metadata: just the bundle state.
  {
    PropertySetRegistry GlobalProps;
    GlobalProps.add(PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
                    static_cast<uint32_t>(State));
    Desc.GlobalMetadata = serializePropSetRegistry(GlobalProps);
  }

  // Each runtime image becomes its own abstract module.
  Desc.AbstractModules.reserve(Images.size());
  for (const RTDeviceBinaryImage *Image : Images) {
    if (!Image)
      continue;
    AbstractModuleDesc &AMD = Desc.AbstractModules.emplace_back();

    // Forward all property sets from the source image into the abstract
    // module metadata. This carries [SYCL/device requirements],
    // [SYCL/specialization constants], etc., verbatim so that compatibility
    // matching at re-load time uses the same predicates.
    {
      PropertySetRegistry AMProps;
      const sycl_device_binary_struct &Raw = Image->getRawData();
      for (sycl_device_binary_property_set PS = Raw.PropertySetsBegin;
           PS != Raw.PropertySetsEnd; ++PS) {
        // Skip SYCLBIN-reserved property sets; those are reconstructed below
        // for IR/native modules and globally above.
        std::string_view Name{PS->Name};
        if (Name == PropertySetRegistry::SYCLBIN_GLOBAL_METADATA ||
            Name == PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA ||
            Name ==
                PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA)
          continue;
        appendPropSetTo(AMProps, PS);
      }
      AMD.Metadata = serializePropSetRegistry(AMProps);
    }

    // Image bytes view.
    const sycl_device_binary_struct &Raw = Image->getRawData();
    std::string_view Bytes{
        reinterpret_cast<const char *>(Raw.BinaryStart),
        static_cast<size_t>(Raw.BinaryEnd - Raw.BinaryStart)};

    const std::string Triple = getTripleFromTargetSpec(Raw.DeviceTargetSpec);

    if (isIRModuleFormat(Image->getFormat())) {
      ImageDesc &ID = AMD.IRModules.emplace_back();
      ID.Bytes = Bytes;
      PropertySetRegistry IRMProps;
      IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type",
                   getIRModuleType(Image->getFormat()));
      IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
                   std::string_view{Triple});
      ID.Metadata = serializePropSetRegistry(IRMProps);
    } else {
      ImageDesc &ID = AMD.NativeDeviceCodeImages.emplace_back();
      ID.Bytes = Bytes;
      PropertySetRegistry NDCIProps;
      // arch is informational; the SYCLBIN reader does not consume it. Forward
      // the compile_target property (already carried in the abstract module
      // metadata above) as the arch string when present, otherwise leave it
      // empty.
      std::string_view Arch{};
      if (sycl_device_binary_property CT =
              Image->getProperty("compile_target")) {
        // BYTE_ARRAY layout: [8-byte size header][payload], with
        // ValSize covering the whole blob. Guard against malformed images
        // whose ValSize is smaller than the mandatory 8-byte header.
        assert(CT->ValSize >= sizeof(PropertyValue::SizeTy) &&
               "compile_target BYTE_ARRAY smaller than its 8-byte header.");
        if (CT->ValSize >= sizeof(PropertyValue::SizeTy)) {
          const char *Bytes = reinterpret_cast<const char *>(CT->ValAddr);
          Arch = std::string_view{Bytes + sizeof(PropertyValue::SizeTy),
                                  CT->ValSize - sizeof(PropertyValue::SizeTy)};
        }
      }
      NDCIProps.add(
          PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
          "arch", Arch);
      NDCIProps.add(
          PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
          "target", std::string_view{Triple});
      ID.Metadata = serializePropSetRegistry(NDCIProps);
    }
  }

  return write(Desc);
}

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
  auto IsDevCompatible = [&](const RTDeviceBinaryImage &Img) {
    return doesDevSupportDeviceRequirements(Dev, Img) &&
           doesImageTargetMatchDevice(Img, Dev);
  };
  auto FindCompatible = [&](const RTDeviceBinaryImage *Imgs, size_t NumImgs) {
    auto *It = std::find_if(Imgs, Imgs + NumImgs, IsDevCompatible);
    return It != Imgs + NumImgs ? It : nullptr;
  };

  std::vector<const RTDeviceBinaryImage *> Images;
  for (size_t I = 0; I < getNumAbstractModules(); ++I) {
    const AbstractModuleDesc &AMDesc = AbstractModuleDescriptors[I];

    // For executable state, native binaries are closest to launch and are
    // preferred. For object/input state, JIT binaries are preferred because
    // ProgramManager::link uses urProgramLinkExp, which requires SPIR-V.
    if (State == bundle_state::executable) {
      if (const RTDeviceBinaryImage *Native =
              FindCompatible(AMDesc.NativeBinaries, AMDesc.NumNativeBinaries)) {
        Images.push_back(Native);
        continue;
      }
    }

    if (const RTDeviceBinaryImage *JIT =
            FindCompatible(AMDesc.JITBinaries, AMDesc.NumJITBinaries)) {
      Images.push_back(JIT);
      continue;
    }

    // No JIT binary available. Native AOT images carry their own notion of
    // state: an AOT image with imported symbols is in object state (link
    // still pending) and an AOT image without imports is already in
    // executable state. Accept the native candidate iff its intrinsic
    // state, as classified by ProgramManager::getBinImageState, matches
    // the requested state. The previous selector skipped this case for
    // non-executable requests, which made AOT-only object SYCLBINs
    // (produced via -fsyclbin=object with an AOT target) load as an empty
    // kernel_bundle and broke any subsequent sycl::link.
    if (const RTDeviceBinaryImage *Native =
            FindCompatible(AMDesc.NativeBinaries, AMDesc.NumNativeBinaries);
        Native && ProgramManager::getBinImageState(Native) == State)
      Images.push_back(Native);
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
