//===- SYCLOffloadWrapper.h --r----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_LINKER_WRAPPER_SYCL_OFFLOAD_WRAPPER_H
#define LLVM_CLANG_TOOLS_CLANG_LINKER_WRAPPER_SYCL_OFFLOAD_WRAPPER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PropertySetIO.h"

#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace offloading {

// SYCL binary image formats supported.
enum class SYCLBinaryImageFormat {
  BIF_None,   // Undetermined Image kind
  BIF_Native, // Native Image kind
  BIF_SPIRV,  // SPIR-V
  BIF_LLVMBC  // LLVM bitcode
};

struct SYCLImage {
  SYCLImage() = default;
  SYCLImage(const SYCLImage &) = delete;
  SYCLImage &operator=(const SYCLImage &) = delete;
  SYCLImage(SYCLImage &&) = default;
  SYCLImage &operator=(SYCLImage &&) = default;

  SYCLImage(std::unique_ptr<llvm::MemoryBuffer> Image,
            const llvm::util::PropertySetRegistry &Registry,
            llvm::StringRef Entries, llvm::StringRef Target = "")
      : Image(std::move(Image)), PropertyRegistry(std::move(Registry)),
        Entries(Entries.begin(), Entries.size()),
        Target(Target.begin(), Target.size()) {}

  std::unique_ptr<llvm::MemoryBuffer> Image;
  llvm::util::PropertySetRegistry PropertyRegistry;

  std::string Entries;

  // Offload target triple.
  std::string Target;

  // Format of the image data - SPIRV, LLVMIR bitcode, etc
  SYCLBinaryImageFormat Format = SYCLBinaryImageFormat::BIF_None;
};

struct SYCLWrappingOptions {
  bool EmitRegistrationFunctions = true;

  // target/compiler specific options what are suggested to use to "compile"
  // program at runtime.
  std::string CompileOptions;
  // Target/Compiler specific options that are suggested to use to "link"
  // program at runtime.
  std::string LinkOptions;
};

/// Wraps the input bundled images and accompanied data into the module \p M
/// as global symbols and registers the images with the SYCL Runtime.
/// \param Options Settings that allows to turn on optional data and settings.
llvm::Error
wrapSYCLBinaries(llvm::Module &M, const llvm::SmallVector<SYCLImage> &Images,
                 SYCLWrappingOptions Options = SYCLWrappingOptions());

} // namespace offloading
} // namespace llvm

#endif
