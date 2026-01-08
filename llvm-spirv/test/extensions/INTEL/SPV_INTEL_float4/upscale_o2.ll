; This test is created to check, if llvm-spirv can work with close-to-real life
; LLVM IR (O2).
; Compiled from:
;typedef half vec2half __attribute__((ext_vector_type(2)));
;
;typedef _BitInt(4) vec2int4 __attribute__((ext_vector_type(2)));
;
;vec2half __builtin_upscale(vec2int4);
;
;kernel void quant_add(local char *in1_ptr, local char *in2_ptr, local vec2half *out_ptr) {
;    int idx = get_global_id(0);
;
;    vec2int4 in1_4bit = (vec2int4)(in1_ptr[idx]);
;    vec2int4 in2_4bit = (vec2int4)(in2_ptr[idx]);
;
;    vec2half in1_upscaled = __builtin_upscale(in1_4bit);
;    vec2half in2_upscaled = __builtin_upscale(in2_4bit);
;
;    out_ptr[idx] = in1_upscaled + in2_upscaled;
;}
;
; with __builtin_upscale function substituted with internal builtin
;
; compile command:
; clang -cl-std=cl3.0 -target spir -emit-llvm -Xclang -finclude-default-header -g0 -O2

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_float4,+SPV_INTEL_int4
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-NOT: _Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i

; CHECK-SPIRV-DAG: Capability Float16Buffer
; CHECK-SPIRV-DAG: Capability Int4TypeINTEL
; CHECK-SPIRV-DAG: Capability Float4E2M1INTEL

; CHECK-SPIRV-DAG: TypeInt [[#Int4Ty:]] 4 0
; CHECK-SPIRV-DAG: TypeVector [[#VecInt4Ty:]] [[#Int4Ty]] 2
; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VecHalfTy:]] [[#HalfTy]] 2
; CHECK-SPIRV-DAG: TypeFloat [[#FP4Ty:]] 4 6214
; CHECK-SPIRV-DAG: TypeVector [[#VecFP4Ty:]] [[#FP4Ty]] 2

; CHECK-SPIRV: Load [[#VecInt4Ty]] [[#VecInt4Val1:]] [[#]] 2 1
; CHECK-SPIRV: Load [[#VecInt4Ty]] [[#VecInt4Val2:]] [[#]] 2 1
; CHECK-SPIRV: Bitcast [[#VecFP4Ty]] [[#Cast1:]] [[#VecInt4Val1]]
; CHECK-SPIRV: FConvert [[#VecHalfTy]] [[#Conv1:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#VecFP4Ty]] [[#Cast2:]] [[#VecInt4Val2]]
; CHECK-SPIRV: FConvert [[#VecHalfTy]] [[#Conv2:]] [[#Cast2]]
; CHECK-SPIRV: FAdd [[#VecHalfTy]] [[#]] [[#Conv1]] [[#Conv2]]

; CHECK-LLVM: %[[#in1:]] = call <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> %[[#]])
; CHECK-LLVM: %[[#in2:]] = call <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> %[[#]])
; CHECK-LLVM: fadd <2 x half> %[[#in1]], %[[#in2]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

define dso_local spir_kernel void @quant_add(ptr addrspace(3) noundef readonly align 1 captures(none) %0, ptr addrspace(3) noundef readonly align 1 captures(none) %1, ptr addrspace(3) noundef writeonly align 4 captures(none) %2) local_unnamed_addr #0 {
  %4 = tail call spir_func i32 @_Z13get_global_idj(i32 noundef 0) #4
  %5 = getelementptr inbounds i8, ptr addrspace(3) %0, i32 %4
  %6 = load <2 x i4>, ptr addrspace(3) %5, align 1
  %7 = getelementptr inbounds i8, ptr addrspace(3) %1, i32 %4
  %8 = load <2 x i4>, ptr addrspace(3) %7, align 1
  %9 = tail call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %6) #5
  %10 = tail call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %8) #5
  %11 = fadd <2 x half> %9, %10
  %12 = getelementptr inbounds <2 x half>, ptr addrspace(3) %2, i32 %4
  store <2 x half> %11, ptr addrspace(3) %12, align 4
  ret void
}

define dso_local spir_func void @__clang_ocl_kern_imp_quant_add(ptr addrspace(3) noundef readonly align 1 captures(none) %0, ptr addrspace(3) noundef readonly align 1 captures(none) %1, ptr addrspace(3) noundef writeonly align 4 captures(none) %2) local_unnamed_addr #1 {
  %4 = tail call spir_func i32 @_Z13get_global_idj(i32 noundef 0) #4
  %5 = getelementptr inbounds i8, ptr addrspace(3) %0, i32 %4
  %6 = load <2 x i4>, ptr addrspace(3) %5, align 1
  %7 = getelementptr inbounds i8, ptr addrspace(3) %1, i32 %4
  %8 = load <2 x i4>, ptr addrspace(3) %7, align 1
  %9 = tail call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %6) #5
  %10 = tail call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %8) #5
  %11 = fadd <2 x half> %9, %10
  %12 = getelementptr inbounds <2 x half>, ptr addrspace(3) %2, i32 %4
  store <2 x half> %11, ptr addrspace(3) %12, align 4
  ret void
}

declare dso_local spir_func i32 @_Z13get_global_idj(i32 noundef) local_unnamed_addr #2

declare dso_local spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef) local_unnamed_addr #3

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { alwaysinline convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { convergent nounwind willreturn memory(none) }
attributes #5 = { convergent nounwind }
