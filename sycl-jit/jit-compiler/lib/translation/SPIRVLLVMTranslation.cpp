//==---------------------- SPIRVLLVMTranslation.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVLLVMTranslation.h"

#include "Kernel.h"
#include "LLVMSPIRVLib.h"
#include "helper/ErrorHandling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace jit_compiler;
using namespace jit_compiler::translation;
using namespace llvm;

SPIRV::TranslatorOpts &SPIRVLLVMTranslator::translatorOpts() {
  // Keep this in sync with clang/lib/Driver/ToolChains/Clang.cpp
  // TODO: consider introducing a config file that both clang and jit-compiler
  // could use during options setting.
  std::vector<SPIRV::ExtensionID> AllowedExtensions{
      SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_add,
      SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_min_max,
      SPIRV::ExtensionID::SPV_KHR_no_integer_wrap_decoration,
      SPIRV::ExtensionID::SPV_KHR_float_controls,
      SPIRV::ExtensionID::SPV_KHR_expect_assume,
      SPIRV::ExtensionID::SPV_KHR_linkonce_odr,
      SPIRV::ExtensionID::SPV_INTEL_subgroups,
      SPIRV::ExtensionID::SPV_INTEL_media_block_io,
      SPIRV::ExtensionID::SPV_INTEL_device_side_avc_motion_estimation,
      SPIRV::ExtensionID::SPV_INTEL_fpga_loop_controls,
      SPIRV::ExtensionID::SPV_INTEL_unstructured_loop_controls,
      SPIRV::ExtensionID::SPV_INTEL_fpga_reg,
      SPIRV::ExtensionID::SPV_INTEL_blocking_pipes,
      SPIRV::ExtensionID::SPV_INTEL_function_pointers,
      SPIRV::ExtensionID::SPV_INTEL_kernel_attributes,
      SPIRV::ExtensionID::SPV_INTEL_io_pipes,
      SPIRV::ExtensionID::SPV_INTEL_inline_assembly,
      SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_integers,
      SPIRV::ExtensionID::SPV_INTEL_float_controls2,
      SPIRV::ExtensionID::SPV_INTEL_vector_compute,
      SPIRV::ExtensionID::SPV_INTEL_fast_composite,
      SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_fixed_point,
      SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_floating_point,
      SPIRV::ExtensionID::SPV_INTEL_variable_length_array,
      SPIRV::ExtensionID::SPV_INTEL_fp_fast_math_mode,
      SPIRV::ExtensionID::SPV_INTEL_long_composites,
      SPIRV::ExtensionID::SPV_INTEL_arithmetic_fence,
      SPIRV::ExtensionID::SPV_INTEL_global_variable_decorations,
      SPIRV::ExtensionID::SPV_INTEL_cache_controls,
      SPIRV::ExtensionID::SPV_INTEL_fpga_buffer_location,
      SPIRV::ExtensionID::SPV_INTEL_fpga_argument_interfaces,
      SPIRV::ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes,
      SPIRV::ExtensionID::SPV_INTEL_fpga_latency_control,
      SPIRV::ExtensionID::SPV_KHR_shader_clock,
      SPIRV::ExtensionID::SPV_INTEL_bindless_images,
      SPIRV::ExtensionID::SPV_INTEL_task_sequence,
      SPIRV::ExtensionID::SPV_INTEL_bfloat16_conversion,
      SPIRV::ExtensionID::SPV_INTEL_joint_matrix,
      SPIRV::ExtensionID::SPV_INTEL_hw_thread_queries,
      SPIRV::ExtensionID::SPV_KHR_uniform_group_instructions,
      SPIRV::ExtensionID::SPV_INTEL_masked_gather_scatter,
      SPIRV::ExtensionID::SPV_INTEL_tensor_float32_conversion,
      SPIRV::ExtensionID::SPV_INTEL_optnone,
      SPIRV::ExtensionID::SPV_KHR_non_semantic_info,
      SPIRV::ExtensionID::SPV_KHR_cooperative_matrix,
      SPIRV::ExtensionID::SPV_EXT_shader_atomic_float16_add,
      SPIRV::ExtensionID::SPV_INTEL_fp_max_error};

  static auto Opts = [&]() -> SPIRV::TranslatorOpts {
    // Options for translation between SPIR-V and LLVM IR.
    // Set SPIRV-V 1.5 as the maximum version number for now.
    // Note that some parts of the code depend on the available builtins, e.g.,
    // passes/kernel-fusion/Builtins.cpp, so updating the SPIR-V version should
    // involve revisiting that code.
    SPIRV::TranslatorOpts TransOpt{SPIRV::VersionNumber::SPIRV_1_5};
    // Enable attachment of kernel arg names as metadata.
    TransOpt.enableGenArgNameMD();
    // Enable mem2reg.
    TransOpt.setMemToRegEnabled(true);
    for (auto &Ext : AllowedExtensions)
      TransOpt.setAllowedToUseExtension(Ext, true);
    TransOpt.setDesiredBIsRepresentation(
        SPIRV::BIsRepresentation::SPIRVFriendlyIR);
    TransOpt.setDebugInfoEIS(
        SPIRV::DebugInfoEIS::NonSemantic_Shader_DebugInfo_200);
    TransOpt.setPreserveAuxData(true);
    const llvm::SmallVector<llvm::StringRef, 4> AllowedIntrinsics = {
        "llvm.genx."};
    TransOpt.setSPIRVAllowUnknownIntrinsics(AllowedIntrinsics);
    // TODO: We need to take care of specialization constants, either by
    // instantiating them by the user-supplied value from the SYCL runtime or by
    // making sure they are correctly represented in the output of the fusion
    // process.
    return TransOpt;
  }();
  return Opts;
}

Expected<std::unique_ptr<llvm::Module>>
SPIRVLLVMTranslator::loadSPIRVKernel(llvm::LLVMContext &LLVMCtx,
                                     SYCLKernelInfo &Kernel) {
  std::unique_ptr<Module> Result{nullptr};

  SYCLKernelBinaryInfo &BinInfo = Kernel.BinaryInfo;
  assert(BinInfo.Format == BinaryFormat::SPIRV &&
         "Only SPIR-V supported as input");

  // Create an input stream for the SPIR-V binary.
  std::stringstream SPIRStream(
      std::string(reinterpret_cast<const char *>(BinInfo.BinaryStart),
                  BinInfo.BinarySize),
      std::ios_base::in | std::ios_base::binary);
  std::string ErrMsg;
  // Create a raw pointer. readSpirv accepts a reference to a pointer,
  // so it will reset the pointer to point to an actual LLVM module.
  Module *LLVMMod;
  auto Success =
      llvm::readSpirv(LLVMCtx, translatorOpts(), SPIRStream, LLVMMod, ErrMsg);
  if (!Success) {
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to load and translate SPIR-V module with error %s",
        ErrMsg.c_str());
  }
  std::unique_ptr<Module> NewMod{LLVMMod};

  return std::move(NewMod);
}

Expected<jit_compiler::KernelBinary *>
SPIRVLLVMTranslator::translateLLVMtoSPIRV(Module &Mod, JITContext &JITCtx) {
  std::ostringstream BinaryStream;
  std::string ErrMsg;
  auto Success = llvm::writeSpirv(&Mod, translatorOpts(), BinaryStream, ErrMsg);
  if (!Success) {
    return createStringError(
        inconvertibleErrorCode(),
        "Translation of LLVM IR to SPIR-V failed with error %s",
        ErrMsg.c_str());
  }
  return &JITCtx.emplaceKernelBinary(BinaryStream.str(), BinaryFormat::SPIRV);
}
