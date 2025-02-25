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
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
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
  
  MemoryBufferRef getMemoryBufferRef() const { return Data; }

  enum class BundleState : uint8_t { Input = 0, Object = 1, Executable = 2 };
  enum class IRType : uint8_t { SPIRV = 0, PTX = 1, AMDGCN = 2 };

  struct ModuleDesc {
    BundleState State;
    std::string ArchString;
    std::vector<module_split::SplitModule> SplitModules;
  };

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t CurrentVersion = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint32_t MagicNumber = 0x53594249;

  /// Serialize the contents of \p ModuleDescs to a binary buffer to be read
  /// later.
  static Expected<SmallString<0>> write(const SmallVector<ModuleDesc> &);

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
    uint64_t ByteTableSize;
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
