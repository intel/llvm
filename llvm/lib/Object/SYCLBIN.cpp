//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace llvm;
using namespace llvm::object;

namespace {

// ---------------------------------------------------------------------------
// V1 (SYBI-magic) reader helpers. Untouched logic from the original
// implementation, retained so files produced by pre-v2 toolchains continue
// to load.
// ---------------------------------------------------------------------------

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

// Parse a serialized PropertySetRegistry blob. PropertySetRegistry::read
// uses line_iterator, which requires a null-terminated MemoryBuffer.
Expected<std::unique_ptr<llvm::util::PropertySetRegistry>>
parsePropertyRegistry(StringRef Blob) {
  auto MemBuf = MemoryBuffer::getMemBufferCopy(Blob);
  return llvm::util::PropertySetRegistry::read(MemBuf.get());
}

// Returns true if Buf starts with the v1 SYBI magic word.
bool hasLegacyMagic(StringRef Buf) {
  if (Buf.size() < sizeof(uint32_t))
    return false;
  uint32_t Magic;
  std::memcpy(&Magic, Buf.data(), sizeof(uint32_t));
  return Magic == SYCLBIN::LegacyMagicNumber;
}

} // namespace

// ---------------------------------------------------------------------------
// SYCLBINDesc construction.
//
// Builds an in-memory description of the SYCLBIN to be written. The on-disk
// layout is decided at write-time by SYCLBIN::write.
// ---------------------------------------------------------------------------

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
      ID.FilePath = SM.ModuleFilePath;
      ID.TargetTripleStr = MD.TargetTriple.str();

      if (MD.ArchString.empty()) {
        // No arch string -> IR module.
        ID.IRType = /*SPIR-V*/ 0;
        AMD.IRModuleDescs.emplace_back(std::move(ID));
      } else {
        // Arch string set -> native device code image.
        ID.ArchString = MD.ArchString;
        AMD.NativeDeviceCodeImageDescs.emplace_back(std::move(ID));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// V2 writer.
//
// The on-disk file is a multi-entry OffloadBinary. Every entry has
// ImageKind == IMG_SYCLBIN, OffloadKind == OFK_SYCL, and conveys its role
// via OffloadBinary StringData keys. Entry image bytes encode property-set
// metadata and (for ir/native entries) the raw binary payload.
// ---------------------------------------------------------------------------

namespace {

struct EntryStorage {
  std::unique_ptr<MemoryBuffer> Image;
  std::string TripleHolder;
  std::string ArchHolder;
  std::string AMIndexHolder;
  std::string IRTypeHolder;
};

} // namespace

// We can't build OffloadingImage directly because StringData values are
// StringRef and need backing storage living until OffloadBinary::write
// returns. The helper writes the file in three passes:
//   1. Build per-entry images (with payload concatenation for ir/native).
//   2. Build OffloadingImage list referencing storage.
//   3. Hand to OffloadBinary::write.
Error SYCLBIN::write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS) {
  // Pass 1: materialize image bytes and per-entry string holders.
  size_t TotalEntries = 1; // global_metadata
  for (const auto &AMD : Desc.AbstractModuleDescs)
    TotalEntries +=
        1 + AMD.IRModuleDescs.size() + AMD.NativeDeviceCodeImageDescs.size();

  SmallVector<EntryStorage> Storage;
  Storage.reserve(TotalEntries);

  SmallVector<OffloadBinary::OffloadingImage> Images;
  Images.reserve(TotalEntries);

  // Helper: build the [u64 metadata_size][metadata][raw_bytes] payload for an
  // ir/native entry.
  auto buildPayload = [](StringRef MetadataBlob, StringRef RawBytes,
                         std::unique_ptr<MemoryBuffer> &Out) -> Error {
    SmallString<0> Buf;
    Buf.reserve(sizeof(uint64_t) + MetadataBlob.size() + RawBytes.size());
    raw_svector_ostream BufOS(Buf);

    uint64_t MetadataSize = MetadataBlob.size();
    char SizeBytes[sizeof(uint64_t)];
    support::endian::write64le(SizeBytes, MetadataSize);
    BufOS.write(SizeBytes, sizeof(SizeBytes));
    BufOS.write(MetadataBlob.data(), MetadataBlob.size());
    BufOS.write(RawBytes.data(), RawBytes.size());

    Out = MemoryBuffer::getMemBufferCopy(Buf);
    return Error::success();
  };

  // Per-image helper: serialize IR/native metadata PropertySetRegistry into
  // a SmallString.
  auto serializeImageMetadata = [](const SYCLBINDesc::ImageDesc &ID,
                                   StringRef Category, SmallString<0> &Out) {
    raw_svector_ostream MetadataOS(Out);
    llvm::util::PropertySetRegistry Reg;
    if (Category ==
        llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA) {
      Reg.add(Category, "type", ID.IRType);
      Reg.add(Category, "target", StringRef(ID.TargetTripleStr));
    } else {
      assert(Category == llvm::util::PropertySetRegistry::
                             SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA);
      Reg.add(Category, "arch", StringRef(ID.ArchString));
      Reg.add(Category, "target", StringRef(ID.TargetTripleStr));
    }
    Reg.write(MetadataOS);
  };

  // Entry 0: global metadata.
  {
    EntryStorage &S = Storage.emplace_back();
    S.Image = MemoryBuffer::getMemBufferCopy(StringRef(Desc.GlobalMetadata));
    OffloadBinary::OffloadingImage Img;
    Img.TheImageKind = IMG_SYCLBIN;
    Img.TheOffloadKind = OFK_SYCL;
    Img.StringData["role"] = "global_metadata";
    Img.Image = nullptr; // patched below after Storage stable
    Images.emplace_back(std::move(Img));
  }

  // Per abstract module: am_metadata, ir entries, native entries.
  for (size_t I = 0; I < Desc.AbstractModuleDescs.size(); ++I) {
    const auto &AMD = Desc.AbstractModuleDescs[I];

    // am_metadata entry.
    {
      EntryStorage &S = Storage.emplace_back();
      S.AMIndexHolder = std::to_string(I);
      S.Image = MemoryBuffer::getMemBufferCopy(StringRef(AMD.Metadata));
      OffloadBinary::OffloadingImage Img;
      Img.TheImageKind = IMG_SYCLBIN;
      Img.TheOffloadKind = OFK_SYCL;
      Img.StringData["role"] = "am_metadata";
      Img.StringData["am_index"] = S.AMIndexHolder;
      Img.Image = nullptr;
      Images.emplace_back(std::move(Img));
    }

    // IR module entries.
    for (const auto &IRMD : AMD.IRModuleDescs) {
      EntryStorage &S = Storage.emplace_back();
      S.AMIndexHolder = std::to_string(I);
      S.IRTypeHolder = std::to_string(IRMD.IRType);
      S.TripleHolder = std::string(IRMD.TargetTripleStr);

      auto FileBufferOrError = MemoryBuffer::getFileOrSTDIN(IRMD.FilePath);
      if (!FileBufferOrError)
        return createFileError(IRMD.FilePath, FileBufferOrError.getError());

      SmallString<0> MetadataBlob;
      serializeImageMetadata(
          IRMD, llvm::util::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA,
          MetadataBlob);

      if (Error E = buildPayload(MetadataBlob,
                                 (*FileBufferOrError)->getBuffer(), S.Image))
        return E;

      OffloadBinary::OffloadingImage Img;
      Img.TheImageKind = IMG_SYCLBIN;
      Img.TheOffloadKind = OFK_SYCL;
      Img.StringData["role"] = "ir";
      Img.StringData["am_index"] = S.AMIndexHolder;
      Img.StringData["ir_type"] = S.IRTypeHolder;
      Img.StringData["triple"] = S.TripleHolder;
      Img.Image = nullptr;
      Images.emplace_back(std::move(Img));
    }

    // Native device-code image entries.
    for (const auto &NDCID : AMD.NativeDeviceCodeImageDescs) {
      EntryStorage &S = Storage.emplace_back();
      S.AMIndexHolder = std::to_string(I);
      S.ArchHolder = std::string(NDCID.ArchString);
      S.TripleHolder = std::string(NDCID.TargetTripleStr);

      auto FileBufferOrError = MemoryBuffer::getFileOrSTDIN(NDCID.FilePath);
      if (!FileBufferOrError)
        return createFileError(NDCID.FilePath, FileBufferOrError.getError());

      SmallString<0> MetadataBlob;
      serializeImageMetadata(NDCID,
                             llvm::util::PropertySetRegistry::
                                 SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
                             MetadataBlob);

      if (Error E = buildPayload(MetadataBlob,
                                 (*FileBufferOrError)->getBuffer(), S.Image))
        return E;

      OffloadBinary::OffloadingImage Img;
      Img.TheImageKind = IMG_SYCLBIN;
      Img.TheOffloadKind = OFK_SYCL;
      Img.StringData["role"] = "native";
      Img.StringData["am_index"] = S.AMIndexHolder;
      Img.StringData["arch"] = S.ArchHolder;
      Img.StringData["triple"] = S.TripleHolder;
      Img.Image = nullptr;
      Images.emplace_back(std::move(Img));
    }
  }

  // Pass 2: patch image pointers to refer to the now-stable Storage entries.
  assert(Storage.size() == Images.size());
  for (size_t I = 0; I < Images.size(); ++I)
    Images[I].Image =
        MemoryBuffer::getMemBuffer(Storage[I].Image->getBuffer(),
                                   /*BufferName=*/"",
                                   /*RequiresNullTerminator=*/false);

  // Pass 3: hand everything to OffloadBinary::write.
  SmallString<0> Bytes = OffloadBinary::write(Images);
  OS.write(Bytes.data(), Bytes.size());
  return Error::success();
}

// ---------------------------------------------------------------------------
// Reader dispatch.
// ---------------------------------------------------------------------------

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::read(MemoryBufferRef Source) {
  StringRef Buf = Source.getBuffer();

  // Bare v1 image: starts with SYBI magic, no OffloadBinary envelope.
  if (hasLegacyMagic(Buf))
    return readV1(Source);

  // Try parsing as an OffloadBinary. Both v1 (single SYBI-image entry) and
  // v2 (multi-entry, no SYBI) are wrapped in an OffloadBinary envelope.
  auto OBVecOrErr = OffloadBinary::create(Source);
  if (!OBVecOrErr) {
    // Not an OffloadBinary and not bare-SYBI -> propagate the parse error.
    return OBVecOrErr.takeError();
  }
  auto &OBVec = *OBVecOrErr;
  if (OBVec.empty())
    return createStringError(inconvertibleErrorCode(),
                             "OffloadBinary contains no entries.");

  // Discriminator: v1 -> first entry's image starts with SYBI magic.
  if (hasLegacyMagic(OBVec[0]->getImage())) {
    // Hand the inner SYBI-magic image to the v1 reader.
    return readV1(
        MemoryBufferRef(OBVec[0]->getImage(), Source.getBufferIdentifier()));
  }

  // v2: dispatch to the multi-entry reader.
  return readV2(Source);
}

// ---------------------------------------------------------------------------
// V1 reader (legacy SYBI-magic path). Behavior preserved.
// ---------------------------------------------------------------------------

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::readV1(MemoryBufferRef Source) {
  auto Result = std::make_unique<SYCLBIN>(Source);
  Result->Version = 1;

  if (Source.getBufferSize() < sizeof(FileHeaderType))
    return createStringError(inconvertibleErrorCode(),
                             "Unexpected file contents size.");

  // Read the file header.
  const FileHeaderType *FileHeader =
      reinterpret_cast<const FileHeaderType *>(Source.getBufferStart());
  if (FileHeader->Magic != LegacyMagicNumber)
    return createStringError(inconvertibleErrorCode(),
                             "Incorrect SYCLBIN magic number.");

  if (FileHeader->Version > 1)
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported legacy SYCLBIN version " +
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

  SYCLBINHeaderBlockReader HeaderBlockReader{Source.getBufferStart(),
                                             HeaderBlockSize};
  SYCLBINByteTableBlockReader MetadataByteTableBlockReader{
      Source.getBufferStart() + HeaderBlockSize,
      static_cast<size_t>(FileHeader->MetadataByteTableSize)};
  SYCLBINByteTableBlockReader BinaryByteTableBlockReader{
      Source.getBufferStart() + HeaderBlockSize + AlignedMetadataByteTableSize,
      static_cast<size_t>(FileHeader->BinaryByteTableSize)};

  if (Error E = MetadataByteTableBlockReader
                    .GetMetadata(FileHeader->GlobalMetadataOffset,
                                 FileHeader->GlobalMetadataSize)
                    .moveInto(Result->GlobalMetadata))
    return std::move(E);

  Result->AbstractModules.resize(FileHeader->AbstractModuleCount);
  for (uint32_t I = 0; I < FileHeader->AbstractModuleCount; ++I) {
    AbstractModule &AM = Result->AbstractModules[I];

    const AbstractModuleHeaderType *AMHeader = nullptr;
    const size_t AMHeaderByteOffset =
        sizeof(FileHeaderType) + sizeof(AbstractModuleHeaderType) * I;
    if (Error E =
            HeaderBlockReader
                .GetHeaderPtr<AbstractModuleHeaderType>(AMHeaderByteOffset)
                .moveInto(AMHeader))
      return std::move(E);

    if (Error E =
            MetadataByteTableBlockReader
                .GetMetadata(AMHeader->MetadataOffset, AMHeader->MetadataSize)
                .moveInto(AM.Metadata))
      return std::move(E);

    AM.IRModules.resize(AMHeader->IRModuleCount);
    for (uint32_t J = 0; J < AMHeader->IRModuleCount; ++J) {
      IRModule &IRM = AM.IRModules[J];

      const IRModuleHeaderType *IRMHeader = nullptr;
      const size_t IRMHeaderByteOffset =
          sizeof(FileHeaderType) + AMHeaderBlockSize +
          sizeof(IRModuleHeaderType) * (AMHeader->IRModuleOffset + J);
      if (Error E = HeaderBlockReader
                        .GetHeaderPtr<IRModuleHeaderType>(IRMHeaderByteOffset)
                        .moveInto(IRMHeader))
        return std::move(E);

      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(IRMHeader->MetadataOffset,
                                     IRMHeader->MetadataSize)
                        .moveInto(IRM.Metadata))
        return std::move(E);

      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(IRMHeader->RawIRBytesOffset,
                                       IRMHeader->RawIRBytesSize)
                        .moveInto(IRM.RawIRBytes))
        return std::move(E);
    }

    AM.NativeDeviceCodeImages.resize(AMHeader->NativeDeviceCodeImageCount);
    for (uint32_t J = 0; J < AMHeader->NativeDeviceCodeImageCount; ++J) {
      NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[J];

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

      if (Error E = MetadataByteTableBlockReader
                        .GetMetadata(NDCIHeader->MetadataOffset,
                                     NDCIHeader->MetadataSize)
                        .moveInto(NDCI.Metadata))
        return std::move(E);

      if (Error E = BinaryByteTableBlockReader
                        .GetBinaryBlob(NDCIHeader->BinaryBytesOffset,
                                       NDCIHeader->BinaryBytesSize)
                        .moveInto(NDCI.RawDeviceCodeImageBytes))
        return std::move(E);
    }
  }

  return std::move(Result);
}

// ---------------------------------------------------------------------------
// V2 reader. Walks the multi-entry OffloadBinary, groups entries by
// am_index, and decodes the [u64 metadata_size][metadata][raw_bytes] payload
// for ir/native entries.
// ---------------------------------------------------------------------------

namespace {

// Decode a u64 little-endian length prefix and split the entry image into
// (metadata_blob, raw_payload).
Error splitImagePayload(StringRef Image, StringRef &Metadata,
                        StringRef &RawBytes) {
  if (Image.size() < sizeof(uint64_t))
    return createStringError(inconvertibleErrorCode(),
                             "SYCLBIN v2 entry image too small.");
  uint64_t MetadataSize = support::endian::read64le(Image.data());
  if (MetadataSize > Image.size() - sizeof(uint64_t))
    return createStringError(inconvertibleErrorCode(),
                             "SYCLBIN v2 entry metadata size out of range.");
  Metadata = Image.substr(sizeof(uint64_t), MetadataSize);
  RawBytes = Image.substr(sizeof(uint64_t) + MetadataSize);
  return Error::success();
}

Expected<uint64_t> parseAMIndex(StringRef S) {
  uint64_t V = 0;
  if (S.empty() || S.getAsInteger(/*Radix=*/10, V))
    return createStringError(inconvertibleErrorCode(),
                             "SYCLBIN v2 invalid am_index '" + S + "'.");
  return V;
}

} // namespace

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::readV2(MemoryBufferRef Source) {
  auto Result = std::make_unique<SYCLBIN>(Source);
  Result->Version = 2;

  auto OBVecOrErr = OffloadBinary::create(Source);
  if (!OBVecOrErr)
    return OBVecOrErr.takeError();
  auto &OBVec = *OBVecOrErr;

  // First pass: locate global metadata, count abstract modules.
  uint64_t MaxAMIndex = 0;
  bool HasAMs = false;
  const OffloadBinary *GlobalMD = nullptr;
  for (const auto &OB : OBVec) {
    if (OB->getImageKind() != IMG_SYCLBIN)
      return createStringError(inconvertibleErrorCode(),
                               "SYCLBIN v2 entry has unexpected ImageKind.");
    StringRef Role = OB->getString("role");
    if (Role == "global_metadata") {
      if (GlobalMD)
        return createStringError(
            inconvertibleErrorCode(),
            "SYCLBIN v2 has multiple global_metadata entries.");
      GlobalMD = OB.get();
    } else if (Role == "am_metadata" || Role == "ir" || Role == "native") {
      auto IdxOrErr = parseAMIndex(OB->getString("am_index"));
      if (!IdxOrErr)
        return IdxOrErr.takeError();
      MaxAMIndex = std::max(MaxAMIndex, *IdxOrErr);
      HasAMs = true;
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "SYCLBIN v2 entry has unknown role '" + Role +
                                   "'.");
    }
  }
  if (!GlobalMD)
    return createStringError(inconvertibleErrorCode(),
                             "SYCLBIN v2 missing global_metadata entry.");

  // Decode global metadata.
  if (Error E = parsePropertyRegistry(GlobalMD->getImage())
                    .moveInto(Result->GlobalMetadata))
    return std::move(E);

  size_t NumAMs = HasAMs ? static_cast<size_t>(MaxAMIndex + 1) : 0;
  Result->AbstractModules.resize(NumAMs);

  // Second pass: populate per-AM metadata, ir, native entries.
  for (const auto &OB : OBVec) {
    StringRef Role = OB->getString("role");
    if (Role == "global_metadata")
      continue;
    auto IdxOrErr = parseAMIndex(OB->getString("am_index"));
    if (!IdxOrErr)
      return IdxOrErr.takeError();
    AbstractModule &AM = Result->AbstractModules[*IdxOrErr];

    if (Role == "am_metadata") {
      if (AM.Metadata)
        return createStringError(
            inconvertibleErrorCode(),
            "SYCLBIN v2 has duplicate am_metadata for am_index " +
                std::to_string(*IdxOrErr) + ".");
      if (Error E = parsePropertyRegistry(OB->getImage()).moveInto(AM.Metadata))
        return std::move(E);
    } else if (Role == "ir" || Role == "native") {
      StringRef MetadataBlob, RawBytes;
      if (Error E = splitImagePayload(OB->getImage(), MetadataBlob, RawBytes))
        return std::move(E);

      std::unique_ptr<llvm::util::PropertySetRegistry> ImgMetadata;
      if (Error E = parsePropertyRegistry(MetadataBlob).moveInto(ImgMetadata))
        return std::move(E);

      if (Role == "ir") {
        IRModule IRM;
        IRM.Metadata = std::move(ImgMetadata);
        IRM.RawIRBytes = RawBytes;
        AM.IRModules.emplace_back(std::move(IRM));
      } else {
        NativeDeviceCodeImage NDCI;
        NDCI.Metadata = std::move(ImgMetadata);
        NDCI.RawDeviceCodeImageBytes = RawBytes;
        AM.NativeDeviceCodeImages.emplace_back(std::move(NDCI));
      }
    }
  }

  // Validate every AM has metadata.
  for (size_t I = 0; I < Result->AbstractModules.size(); ++I) {
    if (!Result->AbstractModules[I].Metadata)
      return createStringError(inconvertibleErrorCode(),
                               "SYCLBIN v2 missing am_metadata for am_index " +
                                   std::to_string(I) + ".");
  }

  return std::move(Result);
}
