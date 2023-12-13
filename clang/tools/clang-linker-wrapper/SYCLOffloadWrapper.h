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
#include "llvm/Support/PropertySetIO.h"

#include <string>
#include <utility>
#include <vector>

// SYCL binary image formats supported.
enum class SYCLBinaryImageFormat {
  BIF_None,   // Undetermined Image kind
  BIF_Native, // Native Image kind
  BIF_SPIRV,  // SPIR-V
  BIF_LLVMBC  // LLVM bitcode
};

SYCLBinaryImageFormat getBinaryImageFormatFromString(llvm::StringRef S);

struct SYCLImage {
  std::vector<char> Image;
  std::optional<llvm::util::PropertySetRegistry> PropertyRegistry;

  // Note: Must be null-terminated.
  std::optional<std::string> Entries;

  // Offload target triple. TODO: check is not performed yet.
  std::string Target;
  // format of the image data - SPIRV, LLVMIR bitcode, etc
  SYCLBinaryImageFormat Format =
      SYCLBinaryImageFormat::BIF_None; // TODO: is not used in Clang Driver.
                                       // legacy
  // target/compiler specific options what are suggested to use to "compile"
  // program at runtime.
  //  Allowed to be empty.
  std::string CompileOptions;
  // Target/Compiler specific options that are suggested to use to "link"
  // program at runtime. Allowed to be empty.
  std::string LinkOptions;
};

SYCLImage getSYCLImage();

struct SYCLWrappingOptions {
  bool EmitRegistrationFunctions = true;
};

llvm::Error
wrapSYCLBinaries(llvm::Module &M, llvm::SmallVector<SYCLImage> &Images,
                 SYCLWrappingOptions Options = SYCLWrappingOptions());

#endif
