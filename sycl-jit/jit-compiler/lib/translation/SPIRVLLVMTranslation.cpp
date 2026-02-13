//===- SPIRVLLVMTranslation.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVLLVMTranslation.h"

#include "LLVMSPIRVLib.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace jit_compiler;
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
    TransOpt.setSPIRVAllowUnknownIntrinsics({"llvm.genx."});
    return TransOpt;
  }();
  return Opts;
}

Expected<jit_compiler::JITBinary *>
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
  return &JITCtx.emplaceBinary(BinaryStream.str(), BinaryFormat::SPIRV);
}
