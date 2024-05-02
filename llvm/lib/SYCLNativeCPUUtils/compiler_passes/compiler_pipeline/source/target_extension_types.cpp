// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/target_extension_types.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <multi_llvm/llvm_version.h>

using namespace compiler::utils;
using namespace llvm;

namespace compiler {
namespace utils {
namespace tgtext {

Type *getEventTy(LLVMContext &Ctx) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return TargetExtType::get(Ctx, "spirv.Event");
#endif
}

Type *getSamplerTy(LLVMContext &Ctx) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return TargetExtType::get(Ctx, "spirv.Sampler");
#endif
}

[[maybe_unused]] static Type *getImageTyHelper(
    LLVMContext &Ctx, ImageTyDimensionalityParam Dim, ImageTyDepthParam Depth,
    ImageTyArrayedParam Arrayed, ImageTyMSParam MS, ImageTySampledParam Sampled,
    ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)Dim;
  (void)Depth;
  (void)Arrayed;
  (void)MS;
  (void)Sampled;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  unsigned IntParams[7];
  IntParams[ImageTyDimensionalityIdx] = Dim;
  IntParams[ImageTyDepthIdx] = Depth;
  IntParams[ImageTyArrayedIdx] = Arrayed;
  IntParams[ImageTyMSIdx] = MS;
  IntParams[ImageTySampledIdx] = Sampled;
  IntParams[ImageTyFormatIdx] = /*Unknown*/ 0;
  IntParams[ImageTyAccessQualIdx] = AccessQual;
  return TargetExtType::get(Ctx, "spirv.Image", Type::getVoidTy(Ctx),
                            IntParams);
#endif
}

[[maybe_unused]] static Type *getOpenCLImageTyHelper(
    LLVMContext &Ctx, ImageTyDimensionalityParam Dim,
    ImageTyArrayedParam Arrayed, ImageTyDepthParam Depth, ImageTyMSParam MS,
    ImageTyAccessQualParam AccessQual) {
  return getImageTyHelper(Ctx, Dim, Depth, Arrayed, MS, ImageSampledRuntime,
                          AccessQual);
}

[[maybe_unused]] static Type *getOpenCLImageTyHelper(
    LLVMContext &Ctx, ImageTyDimensionalityParam Dim,
    ImageTyArrayedParam Arrayed, ImageTyAccessQualParam AccessQual) {
  return getOpenCLImageTyHelper(Ctx, Dim, Arrayed, ImageDepthNone,
                                ImageMSSingleSampled, AccessQual);
}

Type *getImage1DTy(LLVMContext &Ctx, ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(Ctx, ImageDim1D, ImageNonArrayed, AccessQual);
#endif
}

Type *getImage1DArrayTy(LLVMContext &Ctx, ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(Ctx, ImageDim1D, ImageArrayed, AccessQual);
#endif
}

Type *getImage1DBufferTy(LLVMContext &Ctx, ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(Ctx, ImageDimBuffer, ImageNonArrayed,
                                AccessQual);
#endif
}

Type *getImage2DTy(LLVMContext &Ctx, bool Depth, bool MS,
                   ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)Depth;
  (void)MS;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(
      Ctx, ImageDim2D, ImageNonArrayed, Depth ? ImageDepth : ImageDepthNone,
      MS ? ImageMSMultiSampled : ImageMSSingleSampled, AccessQual);
#endif
}

Type *getImage2DArrayTy(LLVMContext &Ctx, bool Depth, bool MS,
                        ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)Depth;
  (void)MS;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(
      Ctx, ImageDim2D, ImageArrayed, Depth ? ImageDepth : ImageDepthNone,
      MS ? ImageMSMultiSampled : ImageMSSingleSampled, AccessQual);
#endif
}

Type *getImage3DTy(LLVMContext &Ctx, ImageTyAccessQualParam AccessQual) {
#if LLVM_VERSION_LESS(17, 0)
  (void)Ctx;
  (void)AccessQual;
  llvm_unreachable("Can't use target extension types before LLVM 17");
#else
  return getOpenCLImageTyHelper(Ctx, ImageDim3D, ImageNonArrayed, AccessQual);
#endif
}

}  // namespace tgtext
}  // namespace utils
}  // namespace compiler
