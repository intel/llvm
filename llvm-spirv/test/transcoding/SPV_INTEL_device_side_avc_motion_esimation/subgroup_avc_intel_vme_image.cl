// RUN: %clang_cc1 -O1 -triple spir-unknown-unknown -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation -o %t.spv
// RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// There is no validation for SPV_INTEL_device_side_avc_motion_estimation implemented in
// SPIRV-Tools. TODO: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-LLVM-COMMON,CHECK-LLVM
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-LLVM-COMMON,CHECK-LLVM-SPIRV
// RUN: llvm-spirv %t.rev.bc --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation -o %t.rev.spv
// RUN: llvm-spirv %t.rev.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable

void foo(__read_only image2d_t src, __read_only image2d_t ref,
         sampler_t sampler, intel_sub_group_avc_ime_payload_t ime_payload,
         intel_sub_group_avc_ime_single_reference_streamin_t sstreamin,
         intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin,
         intel_sub_group_avc_ref_payload_t ref_payload,
         intel_sub_group_avc_sic_payload_t sic_payload) {
  intel_sub_group_avc_ime_evaluate_with_single_reference(src, ref, sampler,
                                                         ime_payload);
  intel_sub_group_avc_ime_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       ime_payload);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streamout(
      src, ref, sampler, ime_payload);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout(
      src, ref, ref, sampler, ime_payload);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streamin(
      src, ref, sampler, ime_payload, sstreamin);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin(
      src, ref, ref, sampler, ime_payload, dstreamin);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streaminout(
      src, ref, sampler, ime_payload, sstreamin);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streaminout(
      src, ref, ref, sampler, ime_payload, dstreamin);

  intel_sub_group_avc_ref_evaluate_with_single_reference(src, ref, sampler,
                                                         ref_payload);
  intel_sub_group_avc_ref_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       ref_payload);
  intel_sub_group_avc_ref_evaluate_with_multi_reference(src, 0, sampler,
                                                        ref_payload);
  intel_sub_group_avc_ref_evaluate_with_multi_reference(src, 0, 0, sampler,
                                                        ref_payload);

  intel_sub_group_avc_sic_evaluate_with_single_reference(src, ref, sampler,
                                                         sic_payload);
  intel_sub_group_avc_sic_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       sic_payload);
  intel_sub_group_avc_sic_evaluate_with_multi_reference(src, 0, sampler,
                                                        sic_payload);
  intel_sub_group_avc_sic_evaluate_with_multi_reference(src, 0, 0, sampler,
                                                        sic_payload);
  intel_sub_group_avc_sic_evaluate_ipe(src, sampler, sic_payload);
}

// CHECK-SPIRV: Capability Groups
// CHECK-SPIRV: Capability SubgroupAvcMotionEstimationINTEL

// CHECK-SPIRV: Extension "SPV_INTEL_device_side_avc_motion_estimation"

// CHECK-SPIRV: TypeImage                                     [[ImageTy:[0-9]+]]
// CHECK-SPIRV: TypeSampler                                   [[SamplerTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImePayloadINTEL                        [[ImePayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeSingleReferenceStreaminINTEL        [[ImeSRefInTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeDualReferenceStreaminINTEL          [[ImeDRefInTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcRefPayloadINTEL                        [[RefPayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcSicPayloadINTEL                        [[SicPayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeVmeImageINTEL                             [[VmeImageTy:[0-9]+]] [[ImageTy]]
// CHECK-SPIRV: TypeAvcImeResultINTEL                         [[ImeResultTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeResultSingleReferenceStreamoutINTEL [[ImeSRefOutTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeResultDualReferenceStreamoutINTEL   [[ImeDRefOutTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcRefResultINTEL                         [[RefResultTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcSicResultINTEL                         [[SicResultTy:[0-9]+]]

// CHECK-LLVM: %[[ImageTy:opencl.image2d_ro_t]] = type opaque
// CHECK-LLVM: %[[SamplerTy:opencl.sampler_t]] = type opaque
// CHECK-LLVM: %[[ImePayloadTy:opencl.intel_sub_group_avc_ime_payload_t]] = type opaque
// CHECK-LLVM: %[[ImeSRefInTy:opencl.intel_sub_group_avc_ime_single_reference_streamin_t]] = type opaque
// CHECK-LLVM: %[[ImeDRefInTy:opencl.intel_sub_group_avc_ime_dual_reference_streamin_t]] = type opaque
// CHECK-LLVM: %[[RefPayloadTy:opencl.intel_sub_group_avc_ref_payload_t]] = type opaque
// CHECK-LLVM: %[[SicPayloadTy:opencl.intel_sub_group_avc_sic_payload_t]] = type opaque
// CHECK-LLVM: %[[ImeResultTy:opencl.intel_sub_group_avc_ime_result_t]] = type opaque
// CHECK-LLVM: %[[ImeSRefOutTy:opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t]] = type opaque
// CHECK-LLVM: %[[ImeDRefOutTy:opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t]] = type opaque
// CHECK-LLVM: %[[RefResultTy:opencl.intel_sub_group_avc_ref_result_t]] = type opaque
// CHECK-LLVM: %[[SicResultTy:opencl.intel_sub_group_avc_sic_result_t]] = type opaque

// CHECK-LLVM-SPIRV: %[[ImageTy:spirv.Image._void_1_0_0_0_0_0_0]] = type opaque
// CHECK-LLVM-SPIRV: %[[SamplerTy:spirv.Sampler]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImePayloadTy:spirv.AvcImePayloadINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImeSRefInTy:spirv.AvcImeSingleReferenceStreaminINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImeDRefInTy:spirv.AvcImeDualReferenceStreaminINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[RefPayloadTy:spirv.AvcRefPayloadINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[SicPayloadTy:spirv.AvcSicPayloadINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[VmeImageTy:spirv.VmeImageINTEL._void_1_0_0_0_0_0_0]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImeResultTy:spirv.AvcImeResultINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImeSRefOutTy:spirv.AvcImeResultSingleReferenceStreamoutINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[ImeDRefOutTy:spirv.AvcImeResultDualReferenceStreamoutINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[RefResultTy:spirv.AvcRefResultINTEL]] = type opaque
// CHECK-LLVM-SPIRV: %[[SicResultTy:spirv.AvcSicResultINTEL]] = type opaque

// CHECK-SPIRV: FunctionParameter [[ImageTy]] [[SrcImg:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[ImageTy]] [[RefImg:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[SamplerTy]] [[Sampler:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[ImePayloadTy]] [[ImePayload:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[ImeSRefInTy]] [[ImeSRefIn:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[ImeDRefInTy]] [[ImeDRefIn:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[RefPayloadTy]] [[RefPayload:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[SicPayloadTy]] [[SicPayload:[0-9]+]]
// CHECK-LLVM-COMMON: @foo(%[[ImageTy]] addrspace(1)* %[[SrcImg:.*]], %[[ImageTy]] addrspace(1)* %[[RefImg:.*]], %[[SamplerTy]] addrspace(2)* %[[Sampler:.*]], %[[ImePayloadTy]]* %[[ImePayload:.*]], %[[ImeSRefInTy]]* %[[ImeSRefIn:.*]], %[[ImeDRefInTy]]* %[[ImeDRefIn:.*]], %[[RefPayloadTy]]* %[[RefPayload:.*]], %[[SicPayloadTy]]* %[[SicPayload:.*]])


// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg0:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg1:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithSingleReferenceINTEL [[ImeResultTy]] {{.*}} [[VmeImg0]] [[VmeImg1]] [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImeResultTy]]* @_Z54intel_sub_group_avc_ime_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg0:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %src, %[[SamplerTy]] addrspace(2)* %sampler)
// CHECK-LLVM-SPIRV: %[[VmeImg1:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %ref, %[[SamplerTy]] addrspace(2)* %sampler)
// CHECK-LLVM-SPIRV: call spir_func %[[ImeResultTy]]* @_Z54__spirv_SubgroupAvcImeEvaluateWithSingleReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTEL(%[[VmeImageTy]] addrspace(1)* %TempSampledImage, %[[VmeImageTy]] addrspace(1)* %[[VmeImg1]], %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg2:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg3:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg4:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithDualReferenceINTEL [[ImeResultTy]] {{.*}} [[VmeImg2]] [[VmeImg3]] [[VmeImg4]] [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImeResultTy]]* @_Z52intel_sub_group_avc_ime_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg2:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg3:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg4:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeResultTy]]* @_Z52__spirv_SubgroupAvcImeEvaluateWithDualReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg2]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg3]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg4]], %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg5:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg6:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL [[ImeSRefOutTy]] {{.*}} [[VmeImg5]] [[VmeImg6]] [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImeSRefOutTy]]* @_Z64intel_sub_group_avc_ime_evaluate_with_single_reference_streamout14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg5:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg6:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeSRefOutTy]]* @_Z63__spirv_SubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg5]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg6]], %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg7:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg8:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg9:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL [[ImeDRefOutTy]] {{.*}} [[VmeImg7]] [[VmeImg8]] [[VmeImg9]] [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImeDRefOutTy]]* @_Z62intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg7:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg8:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg9:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeDRefOutTy]]* @_Z61__spirv_SubgroupAvcImeEvaluateWithDualReferenceStreamoutINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg7]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg8]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg9]], %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg10:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg11:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL [[ImeResultTy]] {{.*}} [[VmeImg10]] [[VmeImg11]] [[ImePayload]] [[ImeSRefIn]]
// CHECK-LLVM: call spir_func %[[ImeResultTy]]* @_Z63intel_sub_group_avc_ime_evaluate_with_single_reference_streamin14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t55ocl_intel_sub_group_avc_ime_single_reference_streamin_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeSRefInTy]]* %[[ImeSRefIn]])
// CHECK-LLVM-SPIRV: %[[VmeImg10:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg11:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeResultTy]]* @_Z62__spirv_SubgroupAvcImeEvaluateWithSingleReferenceStreaminINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTELP42__spirv_AvcImeSingleReferenceStreaminINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg10]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg11]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeSRefInTy]]* %[[ImeSRefIn]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg12:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg13:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg14:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL [[ImeResultTy]] {{.*}} [[VmeImg12]] [[VmeImg13]] [[VmeImg14]] [[ImePayload]] [[ImeDRefIn]]
// CHECK-LLVM: call spir_func %[[ImeResultTy]]* @_Z61intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t53ocl_intel_sub_group_avc_ime_dual_reference_streamin_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeDRefInTy]]* %[[ImeDRefIn]])
// CHECK-LLVM-SPIRV: %[[VmeImg12:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg13:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg14:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeResultTy]]* @_Z60__spirv_SubgroupAvcImeEvaluateWithDualReferenceStreaminINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTELP40__spirv_AvcImeDualReferenceStreaminINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg12]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg13]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg14]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeDRefInTy]]* %[[ImeDRefIn]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg1:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg2:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL [[ImeSRefOutTy]] {{.*}} [[VmeImg1]] [[VmeImg2]] [[ImePayload]] [[ImeSRefIn]]
// CHECK-LLVM: call spir_func %[[ImeSRefOutTy]]* @_Z66intel_sub_group_avc_ime_evaluate_with_single_reference_streaminout14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t55ocl_intel_sub_group_avc_ime_single_reference_streamin_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeSRefInTy]]* %[[ImeSRefIn]])
// CHECK-LLVM-SPIRV: %[[VmeImg15:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg16:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeSRefOutTy]]* @_Z65__spirv_SubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTELP42__spirv_AvcImeSingleReferenceStreaminINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg15]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg16]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeSRefInTy]]* %[[ImeSRefIn]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg1:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg2:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg3:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL [[ImeDRefOutTy]] {{.*}} [[VmeImg1]] [[VmeImg2]] [[VmeImg3]] [[ImePayload]] [[ImeDRefIn]]
// CHECK-LLVM: call spir_func %[[ImeDRefOutTy]]* @_Z64intel_sub_group_avc_ime_evaluate_with_dual_reference_streaminout14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t53ocl_intel_sub_group_avc_ime_dual_reference_streamin_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeDRefInTy]]* %[[ImeDRefIn]])
// CHECK-LLVM-SPIRV: %[[VmeImg17:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg18:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg19:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[ImeDRefOutTy]]* @_Z63__spirv_SubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcImePayloadINTELP40__spirv_AvcImeDualReferenceStreaminINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg17]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg18]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg19]], %[[ImePayloadTy]]* %[[ImePayload]], %[[ImeDRefInTy]]* %[[ImeDRefIn]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg15:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg16:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcRefEvaluateWithSingleReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg15]] [[VmeImg16]] [[RefPayload]]
// CHECK-LLVM: call spir_func %[[RefResultTy]]* @_Z54intel_sub_group_avc_ref_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[RefPayloadTy]]* %[[RefPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg20:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg21:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[RefResultTy]]* @_Z54__spirv_SubgroupAvcRefEvaluateWithSingleReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcRefPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg20]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg21]], %[[RefPayloadTy]]* %[[RefPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg17:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg18:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg19:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcRefEvaluateWithDualReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg17]] [[VmeImg18]] [[VmeImg19]] [[RefPayload]]
// CHECK-LLVM: call spir_func %[[RefResultTy]]* @_Z52intel_sub_group_avc_ref_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[RefPayloadTy]]* %[[RefPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg22:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg23:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg24:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[RefResultTy]]* @_Z52__spirv_SubgroupAvcRefEvaluateWithDualReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcRefPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg22]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg23]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg24]], %[[RefPayloadTy]]* %[[RefPayload]])


// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg20:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcRefEvaluateWithMultiReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg20]] {{.*}} [[RefPayload]]
// CHECK-LLVM: call spir_func %[[RefResultTy]]* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], i32 0, %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[RefPayloadTy]]* %[[RefPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg25:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[RefResultTy]]* @_Z53__spirv_SubgroupAvcRefEvaluateWithMultiReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0iP26__spirv_AvcRefPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg25]], i32 0, %[[RefPayloadTy]]* %[[RefPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg21:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL [[RefResultTy]] {{.*}} [[VmeImg21]] {{.*}} [[RefPayload]]
// CHECK-LLVM: call spir_func %[[RefResultTy]]* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], i32 0, i8 0, %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[RefPayloadTy]]* %[[RefPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg26:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[RefResultTy]]* @_Z63__spirv_SubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0icP26__spirv_AvcRefPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg26]], i32 0, i8 0, %[[RefPayloadTy]]* %[[RefPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg23:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg24:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcSicEvaluateWithSingleReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg23]] [[VmeImg24]] [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicResultTy]]* @_Z54intel_sub_group_avc_sic_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg27:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg28:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[SicResultTy]]* @_Z54__spirv_SubgroupAvcSicEvaluateWithSingleReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcSicPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg27]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg28]], %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg25:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg26:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg27:[0-9]+]] [[RefImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcSicEvaluateWithDualReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg25]] [[VmeImg26]] [[VmeImg27]] [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicResultTy]]* @_Z52intel_sub_group_avc_sic_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg29:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg30:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: %[[VmeImg31:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[RefImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[SicResultTy]]* @_Z52__spirv_SubgroupAvcSicEvaluateWithDualReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0PU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcSicPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg29]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg30]], %[[VmeImageTy]] addrspace(1)* %[[VmeImg31]], %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg28:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcSicEvaluateWithMultiReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg28]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicResultTy]]* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], i32 0, %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg32:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[SicResultTy]]* @_Z53__spirv_SubgroupAvcSicEvaluateWithMultiReferenceINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0iP26__spirv_AvcSicPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg32]], i32 0, %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg29:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL [[SicResultTy]] {{.*}} [[VmeImg29]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicResultTy]]* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], i32 0, i8 0, %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg33:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[SicResultTy]]* @_Z63__spirv_SubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0icP26__spirv_AvcSicPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg33]], i32 0, i8 0, %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV: VmeImageINTEL [[VmeImageTy]] [[VmeImg30:[0-9]+]] [[SrcImg]] [[Sampler]]
// CHECK-SPIRV: SubgroupAvcSicEvaluateIpeINTEL [[SicResultTy]] {{.*}} [[VmeImg30]] [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicResultTy]]* @_Z36intel_sub_group_avc_sic_evaluate_ipe14ocl_image2d_ro11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]], %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-LLVM-SPIRV: %[[VmeImg34:.*]] = call spir_func %[[VmeImageTy]] addrspace(1)* @_Z21__spirv_VmeImageINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%[[ImageTy]] addrspace(1)* %[[SrcImg]], %[[SamplerTy]] addrspace(2)* %[[Sampler]])
// CHECK-LLVM-SPIRV: call spir_func %[[SicResultTy]]* @_Z38__spirv_SubgroupAvcSicEvaluateIpeINTELPU3AS141__spirv_VmeImageINTEL__void_1_0_0_0_0_0_0P26__spirv_AvcSicPayloadINTEL(%[[VmeImageTy]] addrspace(1)* %[[VmeImg34]], %[[SicPayloadTy]]* %[[SicPayload]])
