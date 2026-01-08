//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::object;
using OffloadingImage = OffloadBinary::OffloadingImage;

// Classes in this namespace scope are deprecated and are kept only for backward
// compatibility. They should be removed in the future release.
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
    return llvm::StringRef{Data + ByteOffset, static_cast<size_t>(BlobSize)};
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
  // Write global metadata.
  GlobalMetadata = std::make_unique<llvm::util::PropertySetRegistry>();
  GlobalMetadata->add(llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA,
                      "state", static_cast<uint32_t>(State));

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
      AMD.Metadata =
          std::make_unique<llvm::util::PropertySetRegistry>(SM.Properties);

      ImageDesc ID;
      // Copy the filepath.
      ID.FilePath = SM.ModuleFilePath;
      ID.TargetTriple = MD.TargetTriple;

      // Create metadata and save the descriptor to the right collection.
      if (MD.ArchString.empty()) {
        // If the arch string is empty, it must be an IR module.
        // TODO: Determine type from the input.
        ID.TheImageKind = ImageKind::IMG_SPIRV;
        AMD.IRModuleDescs.emplace_back(std::move(ID));
      } else {
        // If the arch string is not empty, it must be an native device code image.
        ID.TheImageKind = ImageKind::IMG_Object;
        ID.ArchString = MD.ArchString;
        AMD.NativeDeviceCodeImageDescs.emplace_back(std::move(ID));
      }
    }
  }
}

Error SYCLBIN::write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS) {
  SmallVector<OffloadingImage> Images;
  SmallVector<SmallString<128>> Buffers;

  // Write global metadata image.
  OffloadingImage GlobalMDI{};
  GlobalMDI.TheOffloadKind = OffloadKind::OFK_SYCL;
  GlobalMDI.Flags = OIF_NoImage;
  Desc.GlobalMetadata->write(GlobalMDI.StringData, Buffers);
  Images.emplace_back(std::move(GlobalMDI));

  size_t AbstractModuleIndex = 0;
  for (const SYCLBINDesc::AbstractModuleDesc &AMD : Desc.AbstractModuleDescs) {
    SmallString<128> &AbstractModuleID =
        Buffers.emplace_back(std::to_string(AbstractModuleIndex));

    // Store IR modules.
    for (const SYCLBINDesc::ImageDesc &IRMD : AMD.IRModuleDescs) {
      OffloadingImage OI{};

      OI.TheImageKind = IRMD.TheImageKind;
      OI.TheOffloadKind = OffloadKind::OFK_SYCL;

      OI.StringData["syclbin_abstract_module_id"] = AbstractModuleID;
      OI.StringData["triple"] = IRMD.TargetTriple.str();
      AMD.Metadata->write(OI.StringData, Buffers);

      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(IRMD.FilePath);
      if (!FileBufferOrError)
        return createFileError(IRMD.FilePath, FileBufferOrError.getError());
      OI.Image = std::move(*FileBufferOrError);

      Images.emplace_back(std::move(OI));
    }

    // Store native device code images.
    for (const SYCLBINDesc::ImageDesc &NDCID : AMD.NativeDeviceCodeImageDescs) {
      OffloadingImage OI{};

      OI.TheImageKind = NDCID.TheImageKind;
      OI.TheOffloadKind = OffloadKind::OFK_SYCL;

      OI.StringData["syclbin_abstract_module_id"] = AbstractModuleID;
      OI.StringData["triple"] = NDCID.TargetTriple.str();
      OI.StringData["arch"] = NDCID.ArchString;
      AMD.Metadata->write(OI.StringData, Buffers);

      auto FileBufferOrError =
          llvm::MemoryBuffer::getFileOrSTDIN(NDCID.FilePath);
      if (!FileBufferOrError)
        return createFileError(NDCID.FilePath, FileBufferOrError.getError());
      OI.Image = std::move(*FileBufferOrError);

      Images.emplace_back(std::move(OI));
    }

    ++AbstractModuleIndex;
  }

  OS << OffloadBinary::write(Images);
  return Error::success();
}

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::read(MemoryBufferRef Source) {
  // Try to read SYCLBIN in new format (aka augmented OffloadBinary).
  Expected<SmallVector<std::unique_ptr<OffloadBinary>>> OffloadBinariesOrErr =
      OffloadBinary::create(Source);
  if (OffloadBinariesOrErr) {
    if (isSYCLBIN(*OffloadBinariesOrErr))
      return create(std::move(*OffloadBinariesOrErr));

    return createStringError(inconvertibleErrorCode(),
                             "Valid Offload Binary, but not SYCLBIN.");
  }

  // Try to read SYCLBIN in legacy format for backward compatibility
  // After reading, it will be written in OffloadBinary format and read again.
  if (Source.getBufferSize() < sizeof(FileHeaderType))
    return createStringError(inconvertibleErrorCode(),
                             "Unexpected file contents size.");

  // Read the file header.
  const FileHeaderType *FileHeader =
      reinterpret_cast<const FileHeaderType *>(Source.getBufferStart());
  if (FileHeader->Magic != MagicNumber)
    return createStringError(inconvertibleErrorCode(),
                             "Incorrect SYCLBIN magic number.");

  if (FileHeader->Version != 1)
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported SYCLBIN version " +
                                 std::to_string(FileHeader->Version) + ".");

  const size_t AMHeaderBlockSize =
      sizeof(AbstractModuleHeaderType) * FileHeader->AbstractModuleCount;
  const size_t IRMHeaderBlockSize =
      sizeof(IRModuleHeaderType) * FileHeader->IRModuleCount;
  const size_t NDCIHeaderBlockSize = sizeof(NativeDeviceCodeImageHeaderType) *
                                     FileHeader->NativeDeviceCodeImageCount;
  const size_t HeaderBlockSize = sizeof(FileHeaderType) + AMHeaderBlockSize +
                                 IRMHeaderBlockSize + NDCIHeaderBlockSize;
  const size_t AlignedMetadataByteTableSize =
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
      static_cast<size_t>(FileHeader->MetadataByteTableSize)};
  SYCLBINByteTableBlockReader BinaryByteTableBlockReader{
      Source.getBufferStart() + HeaderBlockSize + AlignedMetadataByteTableSize,
      static_cast<size_t>(FileHeader->BinaryByteTableSize)};

  // Read global metadata.
  std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata =
      std::make_unique<llvm::util::PropertySetRegistry>();
  if (Error E = MetadataByteTableBlockReader
                    .GetMetadata(FileHeader->GlobalMetadataOffset,
                                 FileHeader->GlobalMetadataSize)
                    .moveInto(GlobalMetadata))
    return std::move(E);

  SmallVector<OffloadingImage> Images;
  SmallVector<SmallString<128>> Buffers;

  // Write global metadata image.
  OffloadingImage GlobalMDI{};
  GlobalMDI.TheOffloadKind = OffloadKind::OFK_SYCL;
  GlobalMDI.Flags = OIF_NoImage;
  GlobalMetadata->write(GlobalMDI.StringData, Buffers);
  Images.emplace_back(std::move(GlobalMDI));

  // Read the abstract modules.
  size_t AbstractModuleIndex = 0;
  for (uint32_t I = 0; I < FileHeader->AbstractModuleCount; ++I) {
    SmallString<128> &AbstractModuleID =
        Buffers.emplace_back(std::to_string(AbstractModuleIndex));

    // Read the header for the current abstract module.
    const AbstractModuleHeaderType *AMHeader = nullptr;
    const size_t AMHeaderByteOffset =
        sizeof(FileHeaderType) + sizeof(AbstractModuleHeaderType) * I;
    if (Error E =
            HeaderBlockReader
                .GetHeaderPtr<AbstractModuleHeaderType>(AMHeaderByteOffset)
                .moveInto(AMHeader))
      return std::move(E);

    // Read the metadata for the current abstract module.
    std::unique_ptr<llvm::util::PropertySetRegistry> AMMetadata =
        std::make_unique<llvm::util::PropertySetRegistry>();
    if (Error E =
            MetadataByteTableBlockReader
                .GetMetadata(AMHeader->MetadataOffset, AMHeader->MetadataSize)
                .moveInto(AMMetadata))
      return std::move(E);

    // Read the IR modules of the current abstract module.
    for (uint32_t J = 0; J < AMHeader->IRModuleCount; ++J) {
      OffloadingImage OI{};

      // Legacy SYCLBIN format used only SPIR-V image kind.
      OI.TheImageKind = ImageKind::IMG_SPIRV;

      OI.TheOffloadKind = OffloadKind::OFK_SYCL;
      OI.StringData["syclbin_abstract_module_id"] = AbstractModuleID;
      AMMetadata->write(OI.StringData, Buffers);

      // Read the header for the current IR module.
      const IRModuleHeaderType *IRMHeader = nullptr;
      const size_t IRMHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize +
          sizeof(IRModuleHeaderType) * (AMHeader->IRModuleOffset + J);
      if (Error E = HeaderBlockReader
                        .GetHeaderPtr<IRModuleHeaderType>(IRMHeaderByteOffset)
                        .moveInto(IRMHeader))
        return std::move(E);

      // Read the metadata for the current IR module.
      std::unique_ptr<llvm::util::PropertySetRegistry> IRMMetadata =
          std::make_unique<llvm::util::PropertySetRegistry>();
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(IRMHeader->MetadataOffset,
                                     IRMHeader->MetadataSize)
                        .moveInto(IRMMetadata))
        return std::move(E);

      llvm::util::PropertySet PS = (*IRMMetadata)
          [llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA];
      OI.StringData["triple"] = reinterpret_cast<const char *>(
          PS[llvm::util::PropertySet::key_type("target")].asByteArray());

      // Read the binary blob for the current IR module.
      StringRef IRMRawIRBytes;
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(IRMHeader->RawIRBytesOffset,
                                       IRMHeader->RawIRBytesSize)
                        .moveInto(IRMRawIRBytes))
        return std::move(E);
      OI.Image = MemoryBuffer::getMemBuffer(IRMRawIRBytes, "",
                                            /*RequiresNullTerminator=*/false);
      Images.emplace_back(std::move(OI));
    }

    // Read the native device code images of the current abstract module.
    for (uint32_t J = 0; J < AMHeader->NativeDeviceCodeImageCount; ++J) {
      OffloadingImage OI{};

      OI.TheImageKind = ImageKind::IMG_Object;
      OI.TheOffloadKind = OffloadKind::OFK_SYCL;
      OI.StringData["syclbin_abstract_module_id"] = AbstractModuleID;
      AMMetadata->write(OI.StringData, Buffers);

      // Read the header for the current native device code image.
      const NativeDeviceCodeImageHeaderType *NDCIHeader = nullptr;
      const size_t NDCIHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize + IRMHeaderBlockSize +
          sizeof(NativeDeviceCodeImageHeaderType) *
              (AMHeader->NativeDeviceCodeImageOffset + J);
      if (Error E = HeaderBlockReader
                        .GetHeaderPtr<NativeDeviceCodeImageHeaderType>(
                            NDCIHeaderByteOffset)
                        .moveInto(NDCIHeader))
        return std::move(E);

      // Read the metadata for the current native device code image.
      std::unique_ptr<llvm::util::PropertySetRegistry> NDCIMetadata =
          std::make_unique<llvm::util::PropertySetRegistry>();
      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(NDCIHeader->MetadataOffset,
                                     NDCIHeader->MetadataSize)
                        .moveInto(NDCIMetadata))
        return std::move(E);

      llvm::util::PropertySet PS =
          (*NDCIMetadata)[llvm::util::PropertySetRegistry::
                              SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA];
      OI.StringData["triple"] = reinterpret_cast<const char *>(
          PS[llvm::util::PropertySet::key_type("target")].asByteArray());
      OI.StringData["arch"] = reinterpret_cast<const char *>(
          PS[llvm::util::PropertySet::key_type("arch")].asByteArray());

      // Read the binary blob for the current native device code image.
      StringRef NDCIRawDeviceCodeImageBytes;
      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(NDCIHeader->BinaryBytesOffset,
                                       NDCIHeader->BinaryBytesSize)
                        .moveInto(NDCIRawDeviceCodeImageBytes))
        return std::move(E);
      OI.Image = MemoryBuffer::getMemBuffer(NDCIRawDeviceCodeImageBytes, "",
                                            /*RequiresNullTerminator=*/false);
      Images.emplace_back(std::move(OI));
    }
    ++AbstractModuleIndex;
  }

  SmallString<0> NewSYCLBIN = OffloadBinary::write(Images);
  OffloadBinariesOrErr = OffloadBinary::create(
      MemoryBufferRef(NewSYCLBIN, Source.getBufferIdentifier()));
  if (!OffloadBinariesOrErr)
    return std::move(OffloadBinariesOrErr.takeError());

  return create(std::move(*OffloadBinariesOrErr));
}

bool SYCLBIN::isSYCLBIN(
    const SmallVector<std::unique_ptr<OffloadBinary>> &OffloadBinaries) {
  for (const std::unique_ptr<OffloadBinary> &OBPtr : OffloadBinaries) {
    if ((OBPtr->getFlags() & OIF_NoImage) == 0)
      continue;

    StringRef MD = OBPtr->getString(
        llvm::util::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA);
    return !MD.empty();
  }
  return false;
}

Error SYCLBIN::initMetadata() {
  for (const std::unique_ptr<OffloadBinary> &OBPtr : OffloadBinaries){
    if ((OBPtr->getFlags() & OIF_NoImage) == 0) {
      auto ErrorOrProperties =
          llvm::util::PropertySetRegistry::read(OBPtr->strings());
      if (!ErrorOrProperties)
        return ErrorOrProperties.takeError();

      GlobalMetadata = std::move(*ErrorOrProperties);
      continue;
    }

    auto ErrorOrProperties =
        llvm::util::PropertySetRegistry::read(OBPtr->strings());
    if (!ErrorOrProperties)
      return ErrorOrProperties.takeError();

    Metadata[OBPtr.get()] = std::move(*ErrorOrProperties);
  }
  return Error::success();
}

Expected<std::unique_ptr<SYCLBIN>>
SYCLBIN::create(SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries) {
  std::unique_ptr<SYCLBIN> SYCLBINPtr =
      std::unique_ptr<SYCLBIN>(new SYCLBIN(std::move(OffloadBinaries)));

  if (Error E = SYCLBINPtr->initMetadata())
    return std::move(E);

  return SYCLBINPtr;
}
