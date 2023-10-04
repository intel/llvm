//===- OffloadWrapper.h --r-------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_LINKER_WRAPPER_OFFLOAD_WRAPPER_H
#define LLVM_CLANG_TOOLS_CLANG_LINKER_WRAPPER_OFFLOAD_WRAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

/// Wraps the input device images into the module \p M as global symbols and
/// registers the images with the OpenMP Offloading runtime libomptarget.
llvm::Error wrapOpenMPBinaries(llvm::Module &M,
                               llvm::ArrayRef<llvm::ArrayRef<char>> Images);

/// Wraps the input fatbinary image into the module \p M as global symbols and
/// registers the images with the CUDA runtime.
llvm::Error wrapCudaBinary(llvm::Module &M, llvm::ArrayRef<char> Images);

/// Wraps the input bundled image into the module \p M as global symbols and
/// registers the images with the HIP runtime.
llvm::Error wrapHIPBinary(llvm::Module &M, llvm::ArrayRef<char> Images);

// SYCL binary image formats supported.
enum class SYCLBinaryImageFormat {
  BIF_None,   // Undetermined Image kind
  BIF_Native, // Native Image kind
  BIF_SPIRV,  // SPIR-V
  BIF_LLVMBC  // LLVM bitcode
};

SYCLBinaryImageFormat getBinaryImageFormatFromString(llvm::StringRef S);

// TODO: add constructors/utils for creating these images in
// clang-linker-wrapper
struct SYCLImage {
  std::string Image;
  std::optional<llvm::util::PropertySetRegistry> PropertyRegistry;

  // Note: Must be null-terminated.
  std::optional<llvm::StringRef> Entries;

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

// TODO: check whether we need to support -fiopenmp Mode.
llvm::Error
wrapSYCLBinaries(llvm::Module &M, llvm::SmallVector<SYCLImage> &Images,
                 SYCLWrappingOptions Options = SYCLWrappingOptions());

#endif
