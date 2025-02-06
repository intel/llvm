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
  SYCLBIN(MemoryBufferRef Source);

  SYCLBIN(const SYCLBIN &Other) = delete;

  enum class BundleState : uint8_t { Input = 0, Object = 1, Executable = 2 };
  enum class IRType : uint8_t { SPIRV = 0, PTX = 1, AMDGCN = 2 };

  struct ModuleDesc {
    BundleState State;
    std::string ArchString;
    std::vector<module_split::SplitModule> SplitModules;
  };

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t Version = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint8_t MagicNumber[4] = {0x53, 0x59, 0x42, 0x49};

  /// Serialize the contents of \p ModuleDescs to a binary buffer to be read
  /// later.
  static Expected<SmallString<0>> write(const SmallVector<ModuleDesc> &);

  static Expected<std::unique_ptr<SYCLBIN>> read(MemoryBufferRef Source);

  struct IRModule {
    IRType Type;
    SmallVector<char> RawIRBytes;
  };
  struct NativeDeviceCodeImage {
    SmallString<0> ArchString;
    SmallVector<char> RawDeviceCodeImageBytes;
  };

  struct AbstractModule {
    SmallVector<SmallString<0>> KernelNames;
    std::unique_ptr<llvm::util::PropertySetRegistry> Properties;

    SmallVector<IRModule> IRModules;
    SmallVector<NativeDeviceCodeImage> NativeDeviceCodeImages;
  };

  struct {
    uint8_t Magic[4];
    uint32_t Version;
  } Header;

  struct {
    BundleState State;
  } Metadata;

  SmallVector<AbstractModule, 4> AbstractModules;
};

} // namespace object

} // namespace llvm

#endif
