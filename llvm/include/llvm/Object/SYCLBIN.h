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
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {

namespace object {

// Representation of a SYCLBIN binary object. This is intended for use as an
// image inside a OffloadBinary.
class SYCLBIN {
public:
  SYCLBIN(MemoryBufferRef Source) : Data{Source} {}

  SYCLBIN(const SYCLBIN &Other) = delete;
  SYCLBIN(SYCLBIN &&Other) = default;

  SYCLBIN &operator=(const SYCLBIN &Other) = delete;
  SYCLBIN &operator=(SYCLBIN &&Other) = default;

  MemoryBufferRef getMemoryBufferRef() const { return Data; }

  enum class BundleState : uint8_t { Input = 0, Object = 1, Executable = 2 };

  struct SYCLBINModuleDesc {
    std::string ArchString;
    std::vector<module_split::SplitModule> SplitModules;
  };

  class SYCLBINDesc {
  public:
    SYCLBINDesc(BundleState State, ArrayRef<SYCLBINModuleDesc> ModuleDescs);

    SYCLBINDesc(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc(SYCLBINDesc &&Other) = default;

    SYCLBINDesc &operator=(const SYCLBINDesc &Other) = delete;
    SYCLBINDesc &operator=(SYCLBINDesc &&Other) = default;

    size_t getMetadataTableByteSize() const;
    Expected<size_t> getBinaryTableByteSize() const;
    Expected<size_t> getSYCLBINByteSite() const;

  private:
    struct ImageDesc {
      SmallString<0> Metadata;
      SmallString<0> FilePath;
    };

    struct AbstractModuleDesc {
      SmallString<0> Metadata;
      SmallVector<ImageDesc, 4> IRModuleDescs;
      SmallVector<ImageDesc, 4> NativeDeviceCodeImageDescs;
    };

    SmallString<0> GlobalMetadata;
    SmallVector<AbstractModuleDesc, 4> AbstractModuleDescs;

    friend class SYCLBIN;
  };

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t CurrentVersion = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint32_t MagicNumber = 0x53594249;

  /// Serialize \p Desc to \p OS .
  static Error write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS);

  /// Deserialize the contents of \p Source to produce a SYCLBIN object.
  static Expected<std::unique_ptr<SYCLBIN>> read(MemoryBufferRef Source);

  struct IRModule {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    StringRef RawIRBytes;
  };
  struct NativeDeviceCodeImage {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    StringRef RawDeviceCodeImageBytes;
  };

  struct AbstractModule {
    std::unique_ptr<llvm::util::PropertySetRegistry> Metadata;
    SmallVector<IRModule> IRModules;
    SmallVector<NativeDeviceCodeImage> NativeDeviceCodeImages;
  };

  uint32_t Version;
  std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
  SmallVector<AbstractModule, 4> AbstractModules;

private:
  MemoryBufferRef Data;

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
};

} // namespace object

} // namespace llvm

#endif
