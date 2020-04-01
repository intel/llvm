// RUN: %clang_cc1 -O1 -triple spir-unknown-unknown -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation -o %t.spv
// RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// There is no validation for SPV_INTEL_device_side_avc_motion_estimation implemented in
// SPIRV-Tools. TODO: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
void foo(intel_sub_group_avc_ime_payload_t ime_payload,
    intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout,
         intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout,
         intel_sub_group_avc_ime_result_t ime_result,
         intel_sub_group_avc_mce_result_t mce_result,
         intel_sub_group_avc_ref_payload_t ref_payload,
         intel_sub_group_avc_sic_payload_t sic_payload,
         intel_sub_group_avc_sic_result_t sic_result,
         intel_sub_group_avc_mce_payload_t mce_payload) {
  intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_inter_shape_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_intra_luma_shape_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_table(0, 0);
  intel_sub_group_avc_mce_get_default_inter_direction_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_intra_luma_mode_penalty(0, 0);

  intel_sub_group_avc_ime_initialize(0, 0, 0);
  intel_sub_group_avc_ime_set_single_reference(0, 0, ime_payload);
  intel_sub_group_avc_ime_set_dual_reference(0, 0, 0, ime_payload);
  intel_sub_group_avc_ime_ref_window_size(0, 0);
  intel_sub_group_avc_ime_ref_window_size(0, 0);
  intel_sub_group_avc_ime_adjust_ref_offset(0, 0, 0, 0);
  intel_sub_group_avc_ime_set_max_motion_vector_count(0, ime_payload);

  intel_sub_group_avc_ime_get_single_reference_streamin(sstreamout);

  intel_sub_group_avc_ime_get_dual_reference_streamin(dstreamout);

  intel_sub_group_avc_ime_get_border_reached(0i, ime_result);

  intel_sub_group_avc_ime_get_streamout_major_shape_distortions(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_distortions(dstreamout, 0, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors(dstreamout, 0, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids(dstreamout, 0, 0);

  intel_sub_group_avc_ime_set_dual_reference(0, 0, 0, ime_payload);
  intel_sub_group_avc_ime_set_weighted_sad(0, ime_payload);

  intel_sub_group_avc_ime_set_early_search_termination_threshold(0, ime_payload);

  intel_sub_group_avc_fme_initialize(0, 0, 0, 0, 0, 0, 0);
  intel_sub_group_avc_bme_initialize(0, 0, 0, 0, 0, 0, 0, 0);

  intel_sub_group_avc_ref_set_bidirectional_mix_disable(ref_payload);

  intel_sub_group_avc_sic_initialize(0);
  intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, sic_payload);
  intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sic_payload);

  intel_sub_group_avc_sic_configure_skc(0, 0, 0, 0, 0, sic_payload);

  intel_sub_group_avc_sic_set_skc_forward_transform_enable(0, sic_payload);
  intel_sub_group_avc_sic_set_block_based_raw_skip_sad(0, sic_payload);
  intel_sub_group_avc_sic_set_intra_luma_shape_penalty(0, sic_payload);
  intel_sub_group_avc_sic_set_intra_luma_mode_cost_function(0, 0, 0,
                                                            sic_payload);
  intel_sub_group_avc_sic_set_intra_chroma_mode_cost_function(0, sic_payload);

  intel_sub_group_avc_sic_get_best_ipe_luma_distortion(sic_result);
  intel_sub_group_avc_sic_get_motion_vector_mask(0, 0);

  intel_sub_group_avc_mce_set_source_interlaced_field_polarity(0, mce_payload);
  intel_sub_group_avc_mce_set_single_reference_interlaced_field_polarity(
      0, mce_payload);
  intel_sub_group_avc_mce_set_dual_reference_interlaced_field_polarities(
      0, 0, mce_payload);
  intel_sub_group_avc_mce_set_inter_base_multi_reference_penalty(0,
                                                                 mce_payload);
  intel_sub_group_avc_mce_set_inter_shape_penalty(0, mce_payload);
  intel_sub_group_avc_mce_set_inter_direction_penalty(0, mce_payload);
  intel_sub_group_avc_mce_set_motion_vector_cost_function(0, 0, 0, mce_payload);

  intel_sub_group_avc_mce_get_inter_reference_interlaced_field_polarities(
      0, 0, mce_result);
}

// CHECK-SPIRV: Capability Groups
// CHECK-SPIRV: Capability SubgroupAvcMotionEstimationINTEL
// CHECK-SPIRV: Capability SubgroupAvcMotionEstimationIntraINTEL
// CHECK-SPIRV: Capability SubgroupAvcMotionEstimationChromaINTEL
// CHECK-SPIRV: Extension "SPV_INTEL_device_side_avc_motion_estimation"

// CHECK-SPIRV: TypeAvcImePayloadINTEL                        [[ImePayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeResultSingleReferenceStreamoutINTEL [[ImeSRefOutTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeResultDualReferenceStreamoutINTEL   [[ImeDRefOutTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeResultINTEL                         [[ImeResultTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcMceResultINTEL                         [[MceResultTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcRefPayloadINTEL                        [[RefPayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcSicPayloadINTEL                        [[SicPayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcSicResultINTEL                         [[SicResultTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcMcePayloadINTEL                        [[McePayloadTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeSingleReferenceStreaminINTEL        [[ImeSRefInTy:[0-9]+]]
// CHECK-SPIRV: TypeAvcImeDualReferenceStreaminINTEL          [[ImeDRefInTy:[0-9]+]]

// CHECK-LLVM: %[[ImePayloadTy:opencl.intel_sub_group_avc_ime_payload_t]] = type opaque
// CHECK-LLVM: %[[ImeSRefOutTy:opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t]] = type opaque
// CHECK-LLVM: %[[ImeDRefOutTy:opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t]] = type opaque
// CHECK-LLVM: %[[ImeResultTy:opencl.intel_sub_group_avc_ime_result_t]] = type opaque
// CHECK-LLVM: %[[MceResultTy:opencl.intel_sub_group_avc_mce_result_t]] = type opaque
// CHECK-LLVM: %[[RefPayloadTy:opencl.intel_sub_group_avc_ref_payload_t]] = type opaque
// CHECK-LLVM: %[[SicPayloadTy:opencl.intel_sub_group_avc_sic_payload_t]] = type opaque
// CHECK-LLVM: %[[SicResultTy:opencl.intel_sub_group_avc_sic_result_t]] = type opaque
// CHECK-LLVM: %[[McePayloadTy:opencl.intel_sub_group_avc_mce_payload_t]] = type opaque
// CHECK-LLVM: %[[ImeSRefInTy:opencl.intel_sub_group_avc_ime_single_reference_streamin_t]] = type opaque
// CHECK-LLVM: %[[ImeDRefInTy:opencl.intel_sub_group_avc_ime_dual_reference_streamin_t]] = type opaque


// CHECK-SPIRV:  FunctionParameter [[ImePayloadTy]] [[ImePayload:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[ImeSRefOutTy]] [[ImeSRefOut:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[ImeDRefOutTy]] [[ImeDRefOut:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[ImeResultTy]]  [[ImeResult:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[MceResultTy]]  [[MceResult:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[RefPayloadTy]] [[RefPayload:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[SicPayloadTy]] [[SicPayload:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[SicResultTy]]  [[SicResult:[0-9]+]]
// CHECK-SPIRV:  FunctionParameter [[McePayloadTy]] [[McePayload:[0-9]+]]

// CHECK-LLVM: spir_func void @foo(%[[ImePayloadTy]]* %[[ImePayload:.*]], %[[ImeSRefOutTy]]* %[[ImeSRefOut:.*]], %[[ImeDRefOutTy]]* %[[ImeDRefOut:.*]], %[[ImeResultTy]]* %[[ImeResult:.*]], %[[MceResultTy]]* %[[MceResult:.*]], %[[RefPayloadTy]]* %[[RefPayload:.*]], %[[SicPayloadTy]]* %[[SicPayload:.*]], %[[SicResultTy]]* %[[SicResult:.*]], %[[McePayloadTy]]* %[[McePayload:.*]])

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL
// CHECK-LLVM: call spir_func i8 @_Z70intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penaltyhh(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultInterShapePenaltyINTEL
// CHECK-LLVM: call spir_func i64 @_Z55intel_sub_group_avc_mce_get_default_inter_shape_penaltyhh(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL
// CHECK-LLVM: call spir_func i32 @_Z60intel_sub_group_avc_mce_get_default_intra_luma_shape_penaltyhh(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL
// CHECK-LLVM: call spir_func <2 x i32> @_Z66intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_tablehh(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL
// CHECK-LLVM: call spir_func i8 @_Z59intel_sub_group_avc_mce_get_default_inter_direction_penaltyhh(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL
// CHECK-LLVM: call spir_func i8 @_Z59intel_sub_group_avc_mce_get_default_intra_luma_mode_penaltyhh(i8 0, i8 0)


// CHECK-SPIRV:  SubgroupAvcImeInitializeINTEL [[ImePayloadTy]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z34intel_sub_group_avc_ime_initializeDv2_thh(<2 x i16> zeroinitializer, i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcImeSetSingleReferenceINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z44intel_sub_group_avc_ime_set_single_referenceDv2_sh37ocl_intel_sub_group_avc_ime_payload_t(<2 x i16> zeroinitializer, i8 0, %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV:  SubgroupAvcImeSetDualReferenceINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z42intel_sub_group_avc_ime_set_dual_referenceDv2_sS_h37ocl_intel_sub_group_avc_ime_payload_t(<2 x i16> zeroinitializer, <2 x i16> zeroinitializer, i8 0, %[[ImePayloadTy]]* %[[ImePayload]])


// CHECK-SPIRV:  SubgroupAvcImeRefWindowSizeINTEL
// CHECK-LLVM: call spir_func <2 x i16> @_Z39intel_sub_group_avc_ime_ref_window_sizehc(i8 0, i8 0)
// CHECK-SPIRV:  SubgroupAvcImeRefWindowSizeINTEL
// CHECK-LLVM: call spir_func <2 x i16> @_Z39intel_sub_group_avc_ime_ref_window_sizehc(i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcImeAdjustRefOffsetINTEL
// CHECK-LLVM: call spir_func <2 x i16> @_Z41intel_sub_group_avc_ime_adjust_ref_offsetDv2_sDv2_tS0_S0_(<2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer)


// CHECK-SPIRV:  SubgroupAvcImeSetMaxMotionVectorCountINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z51intel_sub_group_avc_ime_set_max_motion_vector_counth37ocl_intel_sub_group_avc_ime_payload_t(i8 0, %[[ImePayloadTy]]* %[[ImePayload]])


// CHECK-SPIRV:  SubgroupAvcImeGetSingleReferenceStreaminINTEL [[ImeSRefInTy]] {{.*}} [[ImeSRefOut]]
// CHECK-LLVM: call spir_func %[[ImeSRefInTy]]* @_Z53intel_sub_group_avc_ime_get_single_reference_streamin63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_t(%[[ImeSRefOutTy]]* %[[ImeSRefOut]])

// CHECK-SPIRV:  SubgroupAvcImeGetDualReferenceStreaminINTEL [[ImeDRefInTy]] {{.*}} [[ImeDRefOut]]
// CHECK-LLVM: call spir_func %[[ImeDRefInTy]]* @_Z51intel_sub_group_avc_ime_get_dual_reference_streamin61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_t(%[[ImeDRefOutTy]]* %[[ImeDRefOut]])

// CHECK-SPIRV:  SubgroupAvcImeGetBorderReachedINTEL {{.*}} [[ImeResult]]
// CHECK-LLVM: call spir_func i8 @_Z42intel_sub_group_avc_ime_get_border_reachedh36ocl_intel_sub_group_avc_ime_result_t(i8 0, %[[ImeResultTy]]* %[[ImeResult]])

// CHECK-SPIRV: SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL {{.*}} [[ImeSRefOut]]
// CHECK-SPIRV: SubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL {{.*}} [[ImeDRefOut]]
// CHECK-SPIRV: SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL {{.*}} [[ImeSRefOut]]
// CHECK-SPIRV: SubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL {{.*}} [[ImeDRefOut]]
// CHECK-SPIRV: SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL {{.*}} [[ImeSRefOut]]
// CHECK-SPIRV: SubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL {{.*}} [[ImeDRefOut]]
// CHECK-LLVM: call spir_func i16 @_Z61intel_sub_group_avc_ime_get_streamout_major_shape_distortions63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_th(%[[ImeSRefOutTy]]* %[[ImeSRefOut]], i8 0)
// CHECK-LLVM: call spir_func i16 @_Z61intel_sub_group_avc_ime_get_streamout_major_shape_distortions61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_thh(%[[ImeDRefOutTy]]* %[[ImeDRefOut]], i8 0, i8 0)
// CHECK-LLVM: call spir_func i32 @_Z64intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_th(%[[ImeSRefOutTy]]* %[[ImeSRefOut]], i8 0)
// CHECK-LLVM: call spir_func i32 @_Z64intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_thh(%[[ImeDRefOutTy]]* %[[ImeDRefOut]], i8 0, i8 0)
// CHECK-LLVM: call spir_func i8 @_Z63intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_th(%[[ImeSRefOutTy]]* %[[ImeSRefOut]], i8 0)
// CHECK-LLVM: call spir_func i8 @_Z63intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_thh(%[[ImeDRefOutTy]]* %[[ImeDRefOut]], i8 0, i8 0)


// CHECK-SPIRV: SubgroupAvcImeSetDualReferenceINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z42intel_sub_group_avc_ime_set_dual_referenceDv2_sS_h37ocl_intel_sub_group_avc_ime_payload_t(<2 x i16> zeroinitializer, <2 x i16> zeroinitializer, i8 0, %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: SubgroupAvcImeSetWeightedSadINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z40intel_sub_group_avc_ime_set_weighted_sadj37ocl_intel_sub_group_avc_ime_payload_t(i32 0, %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV: SubgroupAvcImeSetEarlySearchTerminationThresholdINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
// CHECK-LLVM: call spir_func %[[ImePayloadTy]]* @_Z62intel_sub_group_avc_ime_set_early_search_termination_thresholdh37ocl_intel_sub_group_avc_ime_payload_t(i8 0, %[[ImePayloadTy]]* %[[ImePayload]])

// CHECK-SPIRV:  SubgroupAvcFmeInitializeINTEL [[RefPayloadTy]]
// CHECK-LLVM: call spir_func %[[RefPayloadTy]]* @_Z34intel_sub_group_avc_fme_initializeDv2_tmhhhhh(<2 x i16> zeroinitializer, i64 0, i8 0, i8 0, i8 0, i8 0, i8 0)

// CHECK-SPIRV:  SubgroupAvcBmeInitializeINTEL [[RefPayloadTy]]
// CHECK-LLVM: call spir_func %[[RefPayloadTy]]* @_Z34intel_sub_group_avc_bme_initializeDv2_tmhhhhhh(<2 x i16> zeroinitializer, i64 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0)


// CHECK-SPIRV:  SubgroupAvcRefSetBidirectionalMixDisableINTEL [[RefPayloadTy]] {{.*}} [[RefPayload]]
// CHECK-LLVM: call spir_func %[[RefPayloadTy]]* @_Z53intel_sub_group_avc_ref_set_bidirectional_mix_disable37ocl_intel_sub_group_avc_ref_payload_t(%[[RefPayloadTy]]* %[[RefPayload]])


// CHECK-SPIRV:  SubgroupAvcSicInitializeINTEL [[SicPayloadTy]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z34intel_sub_group_avc_sic_initializeDv2_t(<2 x i16> zeroinitializer)


// CHECK-SPIRV:  SubgroupAvcSicConfigureIpeLumaINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhh37ocl_intel_sub_group_avc_sic_payload_t(i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-SPIRV:  SubgroupAvcSicConfigureIpeLumaChromaINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhttth37ocl_intel_sub_group_avc_sic_payload_t(i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i16 0, i16 0, i16 0, i8 0, %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV: SubgroupAvcSicConfigureSkcINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// call spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z37intel_sub_group_avc_sic_configure_skcjjmhh37ocl_intel_sub_group_avc_sic_payload_t(i32 0, i32 0, i64 0, i8 0, i8 0, %[[SicPayloadTy]]* %[[SicPayload]])
 
// CHECK-SPIRV: SubgroupAvcSicSetSkcForwardTransformEnableINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z56intel_sub_group_avc_sic_set_skc_forward_transform_enablem37ocl_intel_sub_group_avc_sic_payload_t(i64 0, %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-SPIRV: SubgroupAvcSicSetBlockBasedRawSkipSadINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z52intel_sub_group_avc_sic_set_block_based_raw_skip_sadh37ocl_intel_sub_group_avc_sic_payload_t(i8 0, %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-SPIRV: SubgroupAvcSicSetIntraLumaShapePenaltyINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z52intel_sub_group_avc_sic_set_intra_luma_shape_penaltyj37ocl_intel_sub_group_avc_sic_payload_t(i32 0, %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-SPIRV: SubgroupAvcSicSetIntraLumaModeCostFunctionINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z57intel_sub_group_avc_sic_set_intra_luma_mode_cost_functionhjj37ocl_intel_sub_group_avc_sic_payload_t(i8 0, i32 0, i32 0, %[[SicPayloadTy]]* %[[SicPayload]])
// CHECK-SPIRV: SubgroupAvcSicSetIntraChromaModeCostFunctionINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
// CHECK-LLVM: call spir_func %[[SicPayloadTy]]* @_Z59intel_sub_group_avc_sic_set_intra_chroma_mode_cost_functionh37ocl_intel_sub_group_avc_sic_payload_t(i8 0, %[[SicPayloadTy]]* %[[SicPayload]])

// CHECK-SPIRV:  SubgroupAvcSicGetBestIpeLumaDistortionINTEL {{.*}} [[SicResult]]
// CHECK-LLVM: call spir_func i16 @_Z52intel_sub_group_avc_sic_get_best_ipe_luma_distortion36ocl_intel_sub_group_avc_sic_result_t(%[[SicResultTy]]* %[[SicResult]])

// CHECK-SPIRV:  SubgroupAvcSicGetMotionVectorMaskINTEL
// CHECK-LLVM: call spir_func i32 @_Z46intel_sub_group_avc_sic_get_motion_vector_maskjh(i32 0, i8 0)

// CHECK-SPIRV: SubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetInterShapePenaltyINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetInterDirectionPenaltyINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-SPIRV: SubgroupAvcMceSetMotionVectorCostFunctionINTEL [[McePayloadTy]] {{.*}} [[McePayload]]
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z60intel_sub_group_avc_mce_set_source_interlaced_field_polarityh37ocl_intel_sub_group_avc_mce_payload_t(i8 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z70intel_sub_group_avc_mce_set_single_reference_interlaced_field_polarityh37ocl_intel_sub_group_avc_mce_payload_t(i8 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z70intel_sub_group_avc_mce_set_dual_reference_interlaced_field_polaritieshh37ocl_intel_sub_group_avc_mce_payload_t(i8 0, i8 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z62intel_sub_group_avc_mce_set_inter_base_multi_reference_penaltyh37ocl_intel_sub_group_avc_mce_payload_t(i8 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z47intel_sub_group_avc_mce_set_inter_shape_penaltym37ocl_intel_sub_group_avc_mce_payload_t(i64 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z51intel_sub_group_avc_mce_set_inter_direction_penaltyh37ocl_intel_sub_group_avc_mce_payload_t(i8 0, %[[McePayloadTy]]* %[[McePayload]])
// CHECK-LLVM: call spir_func %[[McePayloadTy]]* @_Z55intel_sub_group_avc_mce_set_motion_vector_cost_functionmDv2_jh37ocl_intel_sub_group_avc_mce_payload_t(i64 0, <2 x i32> zeroinitializer, i8 0, %[[McePayloadTy]]* %[[McePayload]])

// CHECK-SPIRV: SubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL {{.*}} [[MceResult]]
