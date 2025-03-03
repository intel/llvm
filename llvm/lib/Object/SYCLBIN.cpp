//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

namespace {

class SYCLBINBlockReader {
protected:
  SYCLBINBlockReader(const char *Data, size_t Size) : Data{Data}, Size{Size} {}

  Error ReadSizeCheck(size_t ByteOffset, size_t ReadSize) {
    if (ByteOffset + ReadSize > Size)
      return createStringError(inconvertibleErrorCode(),
                               "Unexpected file contents size.");
    return Error::success();
  }

  const char *Data = nullptr;
  size_t Size = 0;
};

class SYCLBINHeaderBlockReader : public SYCLBINBlockReader {
public:
  SYCLBINHeaderBlockReader(const char *Data, size_t Size)
      : SYCLBINBlockReader(Data, Size) {}

  template <typename HeaderT>
  Expected<const HeaderT *> GetHeaderPtr(size_t ByteOffset) {
    if (Error ReadSizeError = ReadSizeCheck(ByteOffset, sizeof(HeaderT)))
      return ReadSizeError;
    return reinterpret_cast<const HeaderT *>(Data + ByteOffset);
  }
};

class SYCLBINByteTableBlockReader : public SYCLBINBlockReader {
public:
  SYCLBINByteTableBlockReader(const char *Data, size_t Size)
      : SYCLBINBlockReader(Data, Size) {}

  Expected<StringRef> GetBinaryBlob(size_t ByteOffset, uint64_t BlobSize) {
    if (Error ReadSizeError = ReadSizeCheck(ByteOffset, BlobSize))
      return ReadSizeError;
    return llvm::StringRef{Data + ByteOffset, BlobSize};
  }

  Expected<std::unique_ptr<llvm::util::PropertySetRegistry>>
  GetMetadata(size_t ByteOffset, uint64_t MetadataSize) {
    Expected<StringRef> BlobOrError = GetBinaryBlob(ByteOffset, MetadataSize);
    if (!BlobOrError)
      return BlobOrError.takeError();

    std::string PropStr{BlobOrError->str()};
    auto PropMemBuff = llvm::MemoryBuffer::getMemBuffer(PropStr);
    auto ErrorOrProperties =
        llvm::util::PropertySetRegistry::read(PropMemBuff.get());
    if (!ErrorOrProperties)
      return ErrorOrProperties.takeError();
    return {std::move(*ErrorOrProperties)};
  }
};

} // namespace

Expected<SmallString<0>>
SYCLBIN::write(const SmallVector<SYCLBIN::ModuleDesc> &ModuleDescs) {
  // TODO: Merge by properties, so overlap can live in the same abstract module.

  FileHeaderType FileHeader;
  FileHeader.Magic = MagicNumber;
  FileHeader.Version = CurrentVersion;
  FileHeader.AbstractModuleCount = 0;
  FileHeader.IRModuleCount = 0;
  FileHeader.NativeDeviceCodeImageCount = 0;

  SmallString<0> MetadataByteTable;
  raw_svector_ostream MetadataByteTableOS(MetadataByteTable);
  SmallString<0> BinaryByteTable;
  raw_svector_ostream BinaryByteTableOS(BinaryByteTable);

  {
    // Create global metadata properties.
    FileHeader.GlobalMetadataOffset = MetadataByteTable.size();
    llvm::util::PropertySetRegistry GlobalMetadata;
    // TODO: Determine state from the input.
    GlobalMetadata.add(llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA,
                       "state",
                       /*input*/ 0);
    GlobalMetadata.write(MetadataByteTableOS);
    FileHeader.GlobalMetadataSize =
        MetadataByteTable.size() - FileHeader.GlobalMetadataOffset;
  }

  // Calculate the number of abstract modules.
  for (const ModuleDesc &Desc : ModuleDescs)
    FileHeader.AbstractModuleCount += Desc.SplitModules.size();

  SmallVector<AbstractModuleHeaderType, 4> AMHeaders;
  AMHeaders.reserve(FileHeader.AbstractModuleCount);
  SmallVector<IRModuleHeaderType, 4> IRMHeaders;
  SmallVector<NativeDeviceCodeImageHeaderType, 4> NDCIHeaders;

  for (const ModuleDesc &Desc : ModuleDescs) {
    for (const module_split::SplitModule &SM : Desc.SplitModules) {
      AbstractModuleHeaderType &AMHeader = AMHeaders.emplace_back();
      AMHeader.MetadataOffset = MetadataByteTable.size();
      SM.Properties.write(MetadataByteTableOS);
      AMHeader.MetadataSize =
          MetadataByteTable.size() - AMHeader.MetadataOffset;

      // Read the module data. This is needed no matter what kind of module it
      // is.
      auto BinaryDataOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(SM.ModuleFilePath);
      if (std::error_code EC = BinaryDataOrError.getError())
        return createFileError(SM.ModuleFilePath, EC);
      StringRef RawModuleData{(*BinaryDataOrError)->getBufferStart(),
                              (*BinaryDataOrError)->getBufferSize()};

      // If no arch string is present, the module must be IR.
      AMHeader.IRModuleCount = Desc.ArchString.empty();
      AMHeader.IRModuleOffset = IRMHeaders.size();
      FileHeader.IRModuleCount += AMHeader.IRModuleCount;
      if (AMHeader.IRModuleCount) {
        IRModuleHeaderType &IRMHeader = IRMHeaders.emplace_back();
        {
          // Create metadata properties.
          IRMHeader.MetadataOffset = MetadataByteTable.size();
          llvm::util::PropertySetRegistry IRMMetadata;
          // TODO: Determine state from the input.
          IRMMetadata.add(
              llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA,
              "type",
              /*SPIR-V*/ 0);
          IRMMetadata.write(MetadataByteTableOS);
          IRMHeader.MetadataSize =
              MetadataByteTable.size() - IRMHeader.MetadataOffset;
        }
        IRMHeader.RawIRBytesOffset = BinaryByteTable.size();
        IRMHeader.RawIRBytesSize = RawModuleData.size();
        BinaryByteTableOS << RawModuleData;
      }

      // If arch string is present, the module must be a native device code
      // image.
      AMHeader.NativeDeviceCodeImageCount = !Desc.ArchString.empty();
      AMHeader.NativeDeviceCodeImageOffset = NDCIHeaders.size();
      FileHeader.NativeDeviceCodeImageCount +=
          AMHeader.NativeDeviceCodeImageCount;
      if (AMHeader.NativeDeviceCodeImageCount) {
        NativeDeviceCodeImageHeaderType &NDCIHeader =
            NDCIHeaders.emplace_back();
        {
          // Create metadata properties.
          NDCIHeader.MetadataOffset = MetadataByteTable.size();
          llvm::util::PropertySetRegistry NDCIMetadata;
          NDCIMetadata.add(llvm::util::PropertySetRegistry::
                               SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
                           "arch", Desc.ArchString);
          NDCIMetadata.write(MetadataByteTableOS);
          NDCIHeader.MetadataSize =
              MetadataByteTable.size() - NDCIHeader.MetadataOffset;
        }
        NDCIHeader.BinaryBytesOffset = BinaryByteTable.size();
        NDCIHeader.BinaryBytesSize = RawModuleData.size();
        BinaryByteTableOS << RawModuleData;
      }
    }
  }

  // Record the final byte table sizes.
  FileHeader.MetadataByteTableSize = MetadataByteTable.size();
  FileHeader.BinaryByteTableSize = BinaryByteTable.size();

  // Write to combined data string.
  SmallString<0> Data;
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<const char *>(&FileHeader),
                  sizeof(FileHeaderType));
  OS << StringRef(reinterpret_cast<const char *>(AMHeaders.data()),
                  AMHeaders.size_in_bytes());
  OS << StringRef(reinterpret_cast<const char *>(IRMHeaders.data()),
                  IRMHeaders.size_in_bytes());
  OS << StringRef(reinterpret_cast<const char *>(NDCIHeaders.data()),
                  NDCIHeaders.size_in_bytes());

  // Add metadata byte table and pad to align.
  OS << MetadataByteTable;
  size_t AlignedSize = alignTo(OS.tell(), 8);
  OS.write_zeros(AlignedSize - OS.tell());
  assert(AlignedSize == OS.tell() && "Size mismatch");

  // Add binary byte table and pad to align.
  OS << BinaryByteTable;
  AlignedSize = alignTo(OS.tell(), 8);
  OS.write_zeros(AlignedSize - OS.tell());
  assert(AlignedSize == OS.tell() && "Size mismatch");

  return Data;
}

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::read(MemoryBufferRef Source) {
  auto Result = std::make_unique<SYCLBIN>(Source);

  if (Source.getBufferSize() < sizeof(FileHeaderType))
    return createStringError(inconvertibleErrorCode(),
                             "Unexpected file contents size.");

  // Read the file header.
  const FileHeaderType *FileHeader =
      reinterpret_cast<const FileHeaderType *>(Source.getBufferStart());
  if (FileHeader->Magic != MagicNumber)
    return createStringError(inconvertibleErrorCode(),
                             "Incorrect SYCLBIN magic number.");

  if (FileHeader->Version > CurrentVersion)
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported SYCLBIN version " +
                                 std::to_string(FileHeader->Version) + ".");
  Result->Version = FileHeader->Version;

  const uint64_t AMHeaderBlockSize =
      sizeof(AbstractModuleHeaderType) * FileHeader->AbstractModuleCount;
  const uint64_t IRMHeaderBlockSize =
      sizeof(IRModuleHeaderType) * FileHeader->IRModuleCount;
  const uint64_t NDCIHeaderBlockSize = sizeof(NativeDeviceCodeImageHeaderType) *
                                       FileHeader->NativeDeviceCodeImageCount;
  const uint64_t HeaderBlockSize = sizeof(FileHeaderType) + AMHeaderBlockSize +
                                   IRMHeaderBlockSize + NDCIHeaderBlockSize;
  const uint64_t AlignedMetadataByteTableSize =
      alignTo(FileHeader->MetadataByteTableSize, 8);
  if (Source.getBufferSize() < HeaderBlockSize + AlignedMetadataByteTableSize +
                                   FileHeader->BinaryByteTableSize)
    return createStringError(inconvertibleErrorCode(),
                             "Unexpected file contents size.");

  // Create reader objects. These help with checking out-of-bounds access.
  SYCLBINHeaderBlockReader HeaderBlockReader{Source.getBufferStart(),
                                             HeaderBlockSize};
  SYCLBINByteTableBlockReader MetadataByteTableBlockReader{
      Source.getBufferStart() + HeaderBlockSize,
      FileHeader->MetadataByteTableSize};
  SYCLBINByteTableBlockReader BinaryByteTableBlockReader{
      Source.getBufferStart() + HeaderBlockSize + AlignedMetadataByteTableSize,
      FileHeader->BinaryByteTableSize};

  // Read global metadata.
  if (Error E = MetadataByteTableBlockReader
                    .GetMetadata(FileHeader->GlobalMetadataOffset,
                                 FileHeader->GlobalMetadataSize)
                    .moveInto(Result->GlobalMetadata))
    return E;

  // Read the abstract modules.
  Result->AbstractModules.resize(FileHeader->AbstractModuleCount);
  for (uint32_t I = 0; I < FileHeader->AbstractModuleCount; ++I) {
    AbstractModule &AM = Result->AbstractModules[I];

    // Read the header for the current abstract module.
    const AbstractModuleHeaderType *AMHeader = nullptr;
    const uint64_t AMHeaderByteOffset =
        sizeof(FileHeaderType) + sizeof(AbstractModuleHeaderType) * I;
    if (Error E =
            HeaderBlockReader
                .GetHeaderPtr<AbstractModuleHeaderType>(AMHeaderByteOffset)
                .moveInto(AMHeader))
      return E;

    // Read the metadata for the current abstract module.
    if (Error E =
            MetadataByteTableBlockReader
                .GetMetadata(AMHeader->MetadataOffset, AMHeader->MetadataSize)
                .moveInto(AM.Metadata))
      return E;

    // Read the IR modules of the current abstract module.
    AM.IRModules.resize(AMHeader->IRModuleCount);
    for (uint32_t J = 0; J < AMHeader->IRModuleCount; ++J) {
      IRModule &IRM = AM.IRModules[J];

      // Read the header for the current IR module.
      const IRModuleHeaderType *IRMHeader = nullptr;
      const uint64_t IRMHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize +
          sizeof(IRModuleHeaderType) * (AMHeader->IRModuleOffset + J);
      if (Error E = HeaderBlockReader
                        .GetHeaderPtr<IRModuleHeaderType>(IRMHeaderByteOffset)
                        .moveInto(IRMHeader))
        return E;

      // Read the metadata for the current IR module.
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(IRMHeader->MetadataOffset,
                                     IRMHeader->MetadataSize)
                        .moveInto(IRM.Metadata))
        return E;

      // Read the binary blob for the current IR module.
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(IRMHeader->RawIRBytesOffset,
                                       IRMHeader->RawIRBytesSize)
                        .moveInto(IRM.RawIRBytes))
        return E;
    }

    // Read the native device code images of the current abstract module.
    AM.NativeDeviceCodeImages.resize(AMHeader->NativeDeviceCodeImageCount);
    for (uint32_t J = 0; J < AMHeader->NativeDeviceCodeImageCount; ++J) {
      NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[J];

      // Read the header for the current native device code image.
      const NativeDeviceCodeImageHeaderType *NDCIHeader = nullptr;
      const uint64_t NDCIHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize + IRMHeaderBlockSize +
          sizeof(NativeDeviceCodeImageHeaderType) *
              (AMHeader->NativeDeviceCodeImageOffset + J);
      if (Error E = HeaderBlockReader
                        .GetHeaderPtr<NativeDeviceCodeImageHeaderType>(
                            NDCIHeaderByteOffset)
                        .moveInto(NDCIHeader))
        return E;

      // Read the metadata for the current native device code image.
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(NDCIHeader->MetadataOffset,
                                     NDCIHeader->MetadataSize)
                        .moveInto(NDCI.Metadata))
        return E;

      // Read the binary blob for the current native device code image.
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(NDCIHeader->BinaryBytesOffset,
                                       NDCIHeader->BinaryBytesSize)
                        .moveInto(NDCI.RawDeviceCodeImageBytes))
        return E;
    }
  }

  return std::move(Result);
}
