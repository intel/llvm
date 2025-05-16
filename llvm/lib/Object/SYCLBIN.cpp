//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

SYCLBIN::SYCLBINDesc::SYCLBINDesc(BundleState State,
                                  ArrayRef<SYCLBINModuleDesc> ModuleDescs) {
  // TODO: Merge by properties, so overlap can live in the same abstract module.

  // Write global metadata.
  {
    raw_svector_ostream GlobalMetadataOS(GlobalMetadata);
    llvm::util::PropertySetRegistry GlobalMetadataProps;
    GlobalMetadataProps.add(
        llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
        static_cast<uint32_t>(State));
    GlobalMetadataProps.write(GlobalMetadataOS);
  }

  // We currently create a single abstract module per split module.
  // Some of these should be merged in the future.
  size_t NumAMs = 0;
  for (const SYCLBINModuleDesc &MD : ModuleDescs)
    NumAMs += MD.SplitModules.size();
  AbstractModuleDescs.reserve(NumAMs);

  for (const SYCLBINModuleDesc &MD : ModuleDescs) {
    for (const module_split::SplitModule &SM : MD.SplitModules) {
      AbstractModuleDesc &AMD = AbstractModuleDescs.emplace_back();

      // Write module metadata to the abstract module metadata.
      raw_svector_ostream AMMetadataOS(AMD.Metadata);
      SM.Properties.write(AMMetadataOS);

      ImageDesc ID;
      // Copy the filepath.
      ID.FilePath = SM.ModuleFilePath;

      // Create metadata and save the descriptor to the right collection.
      raw_svector_ostream IDMetadataOS(ID.Metadata);
      if (MD.ArchString.empty()) {
        // If the arch string is empty, it must be an IR module.
        llvm::util::PropertySetRegistry IRMMetadata;
        // TODO: Determine type from the input.
        IRMMetadata.add(
            llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type",
            /*SPIR-V*/ 0);
        IRMMetadata.write(IDMetadataOS);
        AMD.IRModuleDescs.emplace_back(std::move(ID));
      } else {
        // If the arch string is empty, it must be an native device code image.
        llvm::util::PropertySetRegistry NDCIMetadata;
        NDCIMetadata.add(llvm::util::PropertySetRegistry::
                             SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
                         "arch", MD.ArchString);
        NDCIMetadata.write(IDMetadataOS);
        AMD.NativeDeviceCodeImageDescs.emplace_back(std::move(ID));
      }
    }
  }
}

size_t SYCLBIN::SYCLBINDesc::getMetadataTableByteSize() const {
  size_t MetadataTableSize = GlobalMetadata.size();
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : AbstractModuleDescs) {
    MetadataTableSize += AMD.Metadata.size();
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs)
      MetadataTableSize += IRMD.Metadata.size();
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs)
      MetadataTableSize += NDCID.Metadata.size();
  }
  return MetadataTableSize;
}

Expected<size_t> SYCLBIN::SYCLBINDesc::getBinaryTableByteSize() const {
  size_t BinaryTableSize = 0;
  const auto GetFileSizeAndIncrease =
      [&BinaryTableSize](const StringRef FilePath) -> Error {
    uint64_t FileSize = 0;
    if (std::error_code EC = sys::fs::file_size(FilePath, FileSize))
      return createFileError(FilePath, EC);
    BinaryTableSize += FileSize;
    return Error::success();
  };
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : AbstractModuleDescs) {
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs)
      if (Error E = GetFileSizeAndIncrease(IRMD.FilePath))
        return std::move(E);
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs)
      if (Error E = GetFileSizeAndIncrease(NDCID.FilePath))
        return std::move(E);
  }
  return BinaryTableSize;
}

Expected<size_t> SYCLBIN::SYCLBINDesc::getSYCLBINByteSite() const {
  size_t ByteSize = 0;
  ByteSize +=
      alignTo(sizeof(FileHeaderType), 8) +
      alignTo(sizeof(AbstractModuleHeaderType), 8) * AbstractModuleDescs.size();
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : AbstractModuleDescs)
    ByteSize +=
        alignTo(sizeof(IRModuleHeaderType), 8) * AMD.IRModuleDescs.size() +
        alignTo(sizeof(NativeDeviceCodeImageHeaderType), 8) *
            AMD.NativeDeviceCodeImageDescs.size();
  ByteSize += alignTo(getMetadataTableByteSize(), 8);
  size_t BinaryTableSize = 0;
  if (Error E = getBinaryTableByteSize().moveInto(BinaryTableSize))
    return std::move(E);
  ByteSize += alignTo(BinaryTableSize, 8);
  return ByteSize;
}

Error SYCLBIN::write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS) {
  uint32_t IRModuleCount = 0, NativeDeviceCodeImageCount = 0;
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    IRModuleCount += AMD.IRModuleDescs.size();
    NativeDeviceCodeImageCount += AMD.NativeDeviceCodeImageDescs.size();
  }
  size_t HeaderTrackedMetadataOffset = 0, HeaderTrackedBinariesOffset = 0;

  // Headers:
  // Write file header.
  FileHeaderType FileHeader;
  FileHeader.Magic = MagicNumber;
  FileHeader.Version = CurrentVersion;
  FileHeader.AbstractModuleCount = Desc.AbstractModuleDescs.size();
  FileHeader.IRModuleCount = IRModuleCount;
  FileHeader.NativeDeviceCodeImageCount = NativeDeviceCodeImageCount;
  FileHeader.GlobalMetadataOffset = 0;
  FileHeader.GlobalMetadataSize = Desc.GlobalMetadata.size();
  FileHeader.MetadataByteTableSize = Desc.getMetadataTableByteSize();
  if (Error E = Desc.getBinaryTableByteSize().moveInto(
          FileHeader.BinaryByteTableSize))
    return E;
  OS << StringRef(reinterpret_cast<char *>(&FileHeader), sizeof(FileHeader));
  OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());
  HeaderTrackedMetadataOffset += FileHeader.GlobalMetadataSize;

  // Write abstract module headers.
  size_t IRModuleOffset = 0, NativeDeviceCodeImageOffset = 0;
  size_t BinariesCount = 0;
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    AbstractModuleHeaderType AMHeader;
    AMHeader.MetadataOffset = HeaderTrackedMetadataOffset;
    AMHeader.MetadataSize = AMD.Metadata.size();
    AMHeader.IRModuleCount = AMD.IRModuleDescs.size();
    AMHeader.IRModuleOffset = IRModuleOffset;
    AMHeader.NativeDeviceCodeImageCount = AMD.NativeDeviceCodeImageDescs.size();
    AMHeader.NativeDeviceCodeImageOffset = NativeDeviceCodeImageOffset;
    OS << StringRef(reinterpret_cast<char *>(&AMHeader), sizeof(AMHeader));
    OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());
    HeaderTrackedMetadataOffset += AMHeader.MetadataSize;
    BinariesCount +=
        AMHeader.IRModuleCount + AMHeader.NativeDeviceCodeImageCount;
  }

  // Store file handles for later.
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> BinaryFileBuffers;
  BinaryFileBuffers.reserve(BinariesCount);

  // Write IR module headers.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs) {
      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(IRMD.FilePath);
      if (!FileBufferOrError)
        return createFileError(IRMD.FilePath, FileBufferOrError.getError());
      std::unique_ptr<MemoryBuffer> &BFB =
          BinaryFileBuffers.emplace_back(std::move(*FileBufferOrError));

      IRModuleHeaderType IRMHeader;
      IRMHeader.MetadataOffset = HeaderTrackedMetadataOffset;
      IRMHeader.MetadataSize = IRMD.Metadata.size();
      IRMHeader.RawIRBytesOffset = HeaderTrackedBinariesOffset;
      IRMHeader.RawIRBytesSize = BFB->getBufferSize();
      OS << StringRef(reinterpret_cast<char *>(&IRMHeader), sizeof(IRMHeader));
      OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());
      HeaderTrackedMetadataOffset += IRMHeader.MetadataSize;
      HeaderTrackedBinariesOffset += IRMHeader.RawIRBytesSize;
    }
  }

  // Write native device code image headers.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs) {
      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(NDCID.FilePath);
      if (!FileBufferOrError)
        return createFileError(NDCID.FilePath, FileBufferOrError.getError());
      std::unique_ptr<MemoryBuffer> &BFB =
          BinaryFileBuffers.emplace_back(std::move(*FileBufferOrError));

      NativeDeviceCodeImageHeaderType NDCIHeader;
      NDCIHeader.MetadataOffset = HeaderTrackedMetadataOffset;
      NDCIHeader.MetadataSize = NDCID.Metadata.size();
      NDCIHeader.BinaryBytesOffset = HeaderTrackedBinariesOffset;
      NDCIHeader.BinaryBytesSize = BFB->getBufferSize();
      OS << StringRef(reinterpret_cast<char *>(&NDCIHeader),
                      sizeof(NDCIHeader));
      OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());
      HeaderTrackedMetadataOffset += NDCIHeader.MetadataSize;
      HeaderTrackedBinariesOffset += NDCIHeader.BinaryBytesSize;
    }
  }

  // Metadata table:
  // Write global metadata.
  OS << Desc.GlobalMetadata;

  // Write abstract module metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    OS << AMD.Metadata;

  // Write IR module metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs)
      OS << IRMD.Metadata;

  // Write native device code image metadata.
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs)
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs)
      OS << NDCID.Metadata;

  // Pad table to the right alignment.
  OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());

  // Binary byte table:
  for (const std::unique_ptr<MemoryBuffer> &BFB : BinaryFileBuffers)
    OS << StringRef{BFB->getBufferStart(), BFB->getBufferSize()};
  OS.write_zeros(alignTo(OS.tell(), 8) - OS.tell());

  return Error::success();
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
    return std::move(E);

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
      return std::move(E);

    // Read the metadata for the current abstract module.
    if (Error E =
            MetadataByteTableBlockReader
                .GetMetadata(AMHeader->MetadataOffset, AMHeader->MetadataSize)
                .moveInto(AM.Metadata))
      return std::move(E);

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
        return std::move(E);

      // Read the metadata for the current IR module.
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(IRMHeader->MetadataOffset,
                                     IRMHeader->MetadataSize)
                        .moveInto(IRM.Metadata))
        return std::move(E);

      // Read the binary blob for the current IR module.
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(IRMHeader->RawIRBytesOffset,
                                       IRMHeader->RawIRBytesSize)
                        .moveInto(IRM.RawIRBytes))
        return std::move(E);
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
        return std::move(E);

      // Read the metadata for the current native device code image.
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(NDCIHeader->MetadataOffset,
                                     NDCIHeader->MetadataSize)
                        .moveInto(NDCI.Metadata))
        return std::move(E);

      // Read the binary blob for the current native device code image.
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(NDCIHeader->BinaryBytesOffset,
                                       NDCIHeader->BinaryBytesSize)
                        .moveInto(NDCI.RawDeviceCodeImageBytes))
        return std::move(E);
    }
  }

  return std::move(Result);
}
