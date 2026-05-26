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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace module_split {
struct SplitModule;
} // namespace module_split
} // namespace llvm

namespace llvm {

namespace object {

// Representation of a SYCLBIN binary object.
//
// Format versions:
//
//   v1: A SYCL Binary file is an OffloadBinary envelope wrapping a single
//       Entry with ImageKind == IMG_SYCLBIN whose image bytes carry a
//       SYBI-magic header followed by SYCLBIN-private FileHeader,
//       AbstractModuleHeader, IRModuleHeader, NativeDeviceCodeImageHeader
//       tables, a metadata byte table and a binary byte table.
//       Reader retained for backward compatibility with existing files.
//
//   v2: A SYCL Binary file is a multi-entry OffloadBinary. Each Entry has
//       ImageKind == IMG_SYCLBIN and its StringData carries:
//         "role"     : "global_metadata" | "am_metadata" | "ir" | "native"
//         "am_index" : decimal abstract-module index ("ir", "native",
//                      "am_metadata" only)
//         "ir_type"  : decimal IR type tag ("ir" only)
//         "arch"     : architecture name ("native" only)
//         "triple"   : LLVM target triple ("ir", "native" only)
//       The Entry image bytes encode:
//         "global_metadata", "am_metadata": serialized PropertySetRegistry
//         "ir", "native": [u64 little-endian metadata_size][serialized
//                         PropertySetRegistry of metadata_size bytes][raw
//                         IR / native code bytes]
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
      // Set for "ir" entries only.
      uint32_t IRType = 0;
      // Set for "native" entries only.
      SmallString<0> ArchString;
      // Triple as serialized string. Set for "ir" and "native" entries.
      SmallString<0> TargetTripleStr;
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

  /// The current on-disk SYCLBIN format version produced by SYCLBIN::write.
  static constexpr uint32_t CurrentVersion = 2;

  /// Magic number used to identify v1 SYCLBIN images.
  /// The character sequence "SYBI" little-endian. Retained so the v2 reader
  /// can detect and dispatch to the v1 backward-compatibility path.
  static constexpr uint32_t LegacyMagicNumber = 0x53594249;

  /// Serialize \p Desc to \p OS as a v2 multi-entry OffloadBinary.
  static Error write(const SYCLBIN::SYCLBINDesc &Desc, raw_ostream &OS);

  /// Deserialize the contents of \p Source to produce a SYCLBIN object.
  /// Accepts both the v1 and v2 on-disk formats; the v1 reader is retained
  /// for backward compatibility with files produced by older toolchains.
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

  /// On-disk format version that produced this in-memory object. 1 for legacy
  /// SYBI-magic files, 2 for the multi-entry OffloadBinary format.
  uint32_t Version = 0;
  std::unique_ptr<llvm::util::PropertySetRegistry> GlobalMetadata;
  SmallVector<AbstractModule, 4> AbstractModules;

private:
  // Buffer that owns the raw byte storage referenced by every StringRef in
  // AbstractModules. Populated when v1 parsing has to copy decoded property
  // blobs out of an unaligned source; left empty for v2 parses that point
  // directly into Data.
  std::unique_ptr<MemoryBuffer> OwnedStorage;

  MemoryBufferRef Data;

  // Legacy v1 on-disk header types. Retained verbatim so the v1 reader path
  // continues to work for files produced by pre-v2 toolchains.
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

  // Internal readers.
  static Expected<std::unique_ptr<SYCLBIN>> readV1(MemoryBufferRef Source);
  static Expected<std::unique_ptr<SYCLBIN>> readV2(MemoryBufferRef Source);
};

} // namespace object

} // namespace llvm

#endif
