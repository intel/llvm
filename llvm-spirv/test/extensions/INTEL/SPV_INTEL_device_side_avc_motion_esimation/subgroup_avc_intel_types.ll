; Source:
;
; #pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
;
; void foo() {
;   intel_sub_group_avc_mce_payload_t payload_mce; // No literal initializer for mce types
;   intel_sub_group_avc_ime_payload_t payload_ime = CLK_AVC_IME_PAYLOAD_INITIALIZE_INTEL;
;   intel_sub_group_avc_ref_payload_t payload_ref = CLK_AVC_REF_PAYLOAD_INITIALIZE_INTEL;
;   intel_sub_group_avc_sic_payload_t payload_sic = CLK_AVC_SIC_PAYLOAD_INITIALIZE_INTEL;
;
;   intel_sub_group_avc_mce_result_t result_mce; // No literal initializer for mce types
;   intel_sub_group_avc_ime_result_t result_ime = CLK_AVC_IME_RESULT_INITIALIZE_INTEL;
;   intel_sub_group_avc_ref_result_t result_ref = CLK_AVC_REF_RESULT_INITIALIZE_INTEL;
;   intel_sub_group_avc_sic_result_t result_sic = CLK_AVC_SIC_RESULT_INITIALIZE_INTEL;
;
;   intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout = 0x0;
;   intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout = 0x0;
;   intel_sub_group_avc_ime_single_reference_streamin_t sstreamin = 0x0;
;   intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin = 0x0;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation -o - -spirv-text | FileCheck %s

; CHECK: Capability Groups
; CHECK: Capability SubgroupAvcMotionEstimationINTEL
; CHECK: Extension "SPV_INTEL_device_side_avc_motion_estimation"

; CHECK: TypeAvcMcePayloadINTEL
; CHECK: TypeAvcImePayloadINTEL [[IME_PAYLOAD:[0-9]]]
; CHECK: TypeAvcRefPayloadINTEL [[REF_PAYLOAD:[0-9]]]
; CHECK: TypeAvcSicPayloadINTEL [[SIC_PAYLOAD:[0-9]]]
; CHECK: TypeAvcMceResultINTEL
; CHECK: TypeAvcImeResultINTEL [[IME_RESULT:[0-9]]]
; CHECK: TypeAvcRefResultINTEL [[REF_RESULT:[0-9]]]
; CHECK: TypeAvcSicResultINTEL [[SIC_RESULT:[0-9]]]
; CHECK: TypeAvcImeResultSingleReferenceStreamoutINTEL [[SSTREAMOUT:[0-9]]]
; CHECK: TypeAvcImeResultDualReferenceStreamoutINTEL [[DSTREAMOUT:[0-9]]]
; CHECK: TypeAvcImeSingleReferenceStreaminINTEL [[SSTREAMIN:[0-9]]]
; CHECK: TypeAvcImeDualReferenceStreaminINTEL [[DSTREAMIN:[0-9]]]

; CHECK: ConstantNull [[IME_PAYLOAD]]
; CHECK: ConstantNull [[REF_PAYLOAD]]
; CHECK: ConstantNull [[SIC_PAYLOAD]]
; CHECK: ConstantNull [[IME_RESULT]]
; CHECK: ConstantNull [[REF_RESULT]]
; CHECK: ConstantNull [[SIC_RESULT]]
; CHECK: ConstantNull [[SSTREAMOUT]]
; CHECK: ConstantNull [[DSTREAMOUT]]
; CHECK: ConstantNull [[SSTREAMIN]]
; CHECK: ConstantNull [[DSTREAMIN]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  %payload_mce = alloca target("spirv.AvcMcePayloadINTEL"), align 4
  %payload_ime = alloca target("spirv.AvcImePayloadINTEL"), align 4
  %payload_ref = alloca target("spirv.AvcRefPayloadINTEL"), align 4
  %payload_sic = alloca target("spirv.AvcSicPayloadINTEL"), align 4
  %result_mce = alloca target("spirv.AvcMceResultINTEL"), align 4
  %result_ime = alloca target("spirv.AvcImeResultINTEL"), align 4
  %result_ref = alloca target("spirv.AvcRefResultINTEL"), align 4
  %result_sic = alloca target("spirv.AvcSicResultINTEL"), align 4
  %sstreamout = alloca target("spirv.AvcImeResultSingleReferenceStreamoutINTEL"), align 4
  %dstreamout = alloca target("spirv.AvcImeResultDualReferenceStreamoutINTEL"), align 4
  %sstreamin = alloca target("spirv.AvcImeSingleReferenceStreaminINTEL"), align 4
  %dstreamin = alloca target("spirv.AvcImeDualReferenceStreaminINTEL"), align 4
  store target("spirv.AvcImePayloadINTEL") zeroinitializer, ptr %payload_ime, align 4
  store target("spirv.AvcRefPayloadINTEL") zeroinitializer, ptr %payload_ref, align 4
  store target("spirv.AvcSicPayloadINTEL") zeroinitializer, ptr %payload_sic, align 4
  store target("spirv.AvcImeResultINTEL") zeroinitializer, ptr %result_ime, align 4
  store target("spirv.AvcRefResultINTEL") zeroinitializer, ptr %result_ref, align 4
  store target("spirv.AvcSicResultINTEL") zeroinitializer, ptr %result_sic, align 4
  store target("spirv.AvcImeResultSingleReferenceStreamoutINTEL") zeroinitializer, ptr %sstreamout, align 4
  store target("spirv.AvcImeResultDualReferenceStreamoutINTEL") zeroinitializer, ptr %dstreamout, align 4
  store target("spirv.AvcImeSingleReferenceStreaminINTEL") zeroinitializer, ptr %sstreamin, align 4
  store target("spirv.AvcImeDualReferenceStreaminINTEL") zeroinitializer, ptr %dstreamin, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{}
!3 = !{!"clang version 5.0.1 (cfe/trunk)"}
