//===- SYCLBIN.h - SYCLBIN binary format support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SYCLBIN_H
#define LLVM_OBJECT_SYCLBIN_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {

namespace object {

// Representation of a SYCLBIN binary object.
// Currently SYCLBIN doesn't own memory, so user must ensure memory used to
// initialize SYCLBIN remains valid for the lifetime of the SYCLBIN object. So
// far I did not find use case, where we want to move memory ownership to
// SYCLBIN or do a memory copy. This can be changed if we find this use case.
class SYCLBIN {
public:
  SYCLBIN(const SYCLBIN &Other) = delete;
  SYCLBIN &operator=(const SYCLBIN &Other) = delete;

  enum class BundleState : uint8_t { Input = 0, Object = 1, Executable = 2 };

  struct SYCLBINModuleDesc {
    std::string ArchString;
    llvm::Triple TargetTriple;
    std::vector<module_split::SplitModule> SplitModules;
  };

  class SYCLBINDesc {
  public:
    SYCLBINDesc(BundleState State, ArrayRef<SYCLBINModuleDesc> ModuleDescs);

    SYCLBINDesc(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc(SYCLBINDesc &&Other) = default;

    SYCLBINDesc &operator=(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc &operator=(SYCLBINDesc &&Other) = default;

  private:
    struct ImageDesc {
      ImageKind TheImageKind = ImageKind::IMG_None;
      llvm::Triple TargetTriple;
      std::string ArchString;
      SmallString<0> FilePath;
    };

    struct AbstractModuleDesc {
      std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
      SmallVector<ImageDesc, 4> IRModuleDescs;
      SmallVector<ImageDesc, 4> NativeDeviceCodeImageDescs;
    };

    std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
    SmallVector<AbstractModuleDesc, 4> AbstractModuleDescs;

    friend class SYCLBIN;
  };

  /// Create a SYCLBIN object from a vector of OffloadBinary objects.
  static Expected<std::unique_ptr<SYCLBIN>>
  create(SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries);

  /// Serialize \p Desc.
  static Error write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS);

  /// Deserialize the contents of \p Source to produce a SYCLBIN object.
  static Expected<std::unique_ptr<SYCLBIN>> read(MemoryBufferRef Source);

  /// Check if OffloadBinary is a SYCLBIN.
  static bool
  isSYCLBIN(const SmallVector<std::unique_ptr<OffloadBinary>> &OffloadBinaries);

  ArrayRef<std::unique_ptr<OffloadBinary>> getOffloadBinaries() const {
    return OffloadBinaries;
  }

  std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
  DenseMap<const OffloadBinary *,
           std::unique_ptr<llvm::util::PropertySetRegistry>>
      Metadata;

private:
  SYCLBIN(SmallVector<std::unique_ptr<OffloadBinary>> OB)
      : OffloadBinaries(std::move(OB)) {}
  
  Error initMetadata();

  SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries;

  // The types and fields below are kept for backward compatibility and should
  // be removed in the future:

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint32_t MagicNumber = 0x53594249;

  struct alignas(8) FileHeaderType {
    uint32_t Magic;
    uint32_t Version;
    uint32_t AbstractModuleCount;
    uint32_t IRModuleCount;
    uint32_t NativeDeviceCodeImageCount;
    uint64_t MetadataByteTableSize;
    uint64_t BinaryByteTableSize;
    uint64_t GlobalMetadataOffset;
    uint64_t GlobalMetadataSize;
  };

  struct alignas(8) AbstractModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint32_t IRModuleCount;
    uint32_t IRModuleOffset;
    uint32_t NativeDeviceCodeImageCount;
    uint32_t NativeDeviceCodeImageOffset;
  };

  struct alignas(8) IRModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t RawIRBytesOffset;
    uint64_t RawIRBytesSize;
  };

  struct alignas(8) NativeDeviceCodeImageHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t BinaryBytesOffset;
    uint64_t BinaryBytesSize;
  };

  // End of deprecated types and fields.
};

} // namespace object

} // namespace llvm

#endif
