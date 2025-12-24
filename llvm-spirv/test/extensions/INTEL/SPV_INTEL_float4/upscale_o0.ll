; This test is created to check, if llvm-spirv can work with close-to-real life
; LLVM IR (O0).
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
; clang -cl-std=cl3.0 -target spir -emit-llvm -Xclang -finclude-default-header -g0 -O0

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
; CHECK-SPIRV-DAG: TypePointer [[#PtrVecInt4Ty:]] 7 [[#VecInt4Ty]]
; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VecHalfTy:]] [[#HalfTy]] 2
; CHECK-SPIRV-DAG: TypeFloat [[#FP4Ty:]] 4 6214
; CHECK-SPIRV-DAG: TypeVector [[#VecFP4Ty:]] [[#FP4Ty]] 2

; CHECK-SPIRV: Load [[#VecInt4Ty]] [[#VecInt4Val1:]] [[#]] 2 2
; CHECK-SPIRV: Bitcast [[#VecFP4Ty]] [[#Cast1:]] [[#VecInt4Val1]]
; CHECK-SPIRV: FConvert [[#VecHalfTy]] [[#Conv1:]] [[#Cast1]]
; CHECK-SPIRV: Store [[#]] [[#Conv1]] 2 4

; CHECK-SPIRV: Load [[#VecInt4Ty]] [[#VecInt4Val2:]] [[#]] 2 2
; CHECK-SPIRV: Bitcast [[#VecFP4Ty]] [[#Cast2:]] [[#VecInt4Val2]]
; CHECK-SPIRV: FConvert [[#VecHalfTy]] [[#Conv2:]] [[#Cast2]]
; CHECK-SPIRV: Store [[#]] [[#Conv2]] 2 4

; CHECK-LLVM: %[[#Conv1:]] = call <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> %[[#]])
; CHECK-LLVM: store <2 x half> %[[#Conv1]], ptr %[[#]]
; CHECK-LLVM: %[[#Conv2:]] = call <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> %[[#]])
; CHECK-LLVM: store <2 x half> %[[#Conv2]], ptr %[[#]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

define dso_local spir_kernel void @quant_add(ptr addrspace(3) noundef align 1 %0, ptr addrspace(3) noundef align 1 %1, ptr addrspace(3) noundef align 4 %2) #0 {
  %4 = alloca ptr addrspace(3), align 4
  %5 = alloca ptr addrspace(3), align 4
  %6 = alloca ptr addrspace(3), align 4
  store ptr addrspace(3) %0, ptr %4, align 4
  store ptr addrspace(3) %1, ptr %5, align 4
  store ptr addrspace(3) %2, ptr %6, align 4
  %7 = load ptr addrspace(3), ptr %4, align 4
  %8 = load ptr addrspace(3), ptr %5, align 4
  %9 = load ptr addrspace(3), ptr %6, align 4
  call spir_func void @__clang_ocl_kern_imp_quant_add(ptr addrspace(3) noundef align 1 %7, ptr addrspace(3) noundef align 1 %8, ptr addrspace(3) noundef align 4 %9) #3
  ret void
}

define dso_local spir_func void @__clang_ocl_kern_imp_quant_add(ptr addrspace(3) noundef align 1 %0, ptr addrspace(3) noundef align 1 %1, ptr addrspace(3) noundef align 4 %2) #0 {
  %4 = alloca ptr addrspace(3), align 4
  %5 = alloca ptr addrspace(3), align 4
  %6 = alloca ptr addrspace(3), align 4
  %7 = alloca i32, align 4
  %8 = alloca <2 x i4>, align 2
  %9 = alloca <2 x i4>, align 2
  %10 = alloca <2 x half>, align 4
  %11 = alloca <2 x half>, align 4
  store ptr addrspace(3) %0, ptr %4, align 4
  store ptr addrspace(3) %1, ptr %5, align 4
  store ptr addrspace(3) %2, ptr %6, align 4
  %12 = call spir_func i32 @_Z13get_global_idj(i32 noundef 0) #4
  store i32 %12, ptr %7, align 4
  %13 = load ptr addrspace(3), ptr %4, align 4
  %14 = load i32, ptr %7, align 4
  %15 = getelementptr inbounds i8, ptr addrspace(3) %13, i32 %14
  %16 = load i8, ptr addrspace(3) %15, align 1
  %17 = trunc i8 %16 to i4
  %18 = insertelement <2 x i4> poison, i4 %17, i64 0
  %19 = shufflevector <2 x i4> %18, <2 x i4> poison, <2 x i32> zeroinitializer
  store <2 x i4> %19, ptr %8, align 2
  %20 = load ptr addrspace(3), ptr %5, align 4
  %21 = load i32, ptr %7, align 4
  %22 = getelementptr inbounds i8, ptr addrspace(3) %20, i32 %21
  %23 = load i8, ptr addrspace(3) %22, align 1
  %24 = trunc i8 %23 to i4
  %25 = insertelement <2 x i4> poison, i4 %24, i64 0
  %26 = shufflevector <2 x i4> %25, <2 x i4> poison, <2 x i32> zeroinitializer
  store <2 x i4> %26, ptr %9, align 2
  %27 = load <2 x i4>, ptr %8, align 2
  %28 = call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %27) #5
  store <2 x half> %28, ptr %10, align 4
  %29 = load <2 x i4>, ptr %9, align 2
  %30 = call spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef %29) #5
  store <2 x half> %30, ptr %11, align 4
  %31 = load <2 x half>, ptr %10, align 4
  %32 = load <2 x half>, ptr %11, align 4
  %33 = fadd <2 x half> %31, %32
  %34 = load ptr addrspace(3), ptr %6, align 4
  %35 = load i32, ptr %7, align 4
  %36 = getelementptr inbounds <2 x half>, ptr addrspace(3) %34, i32 %35
  store <2 x half> %33, ptr addrspace(3) %36, align 4
  ret void
}

declare dso_local spir_func i32 @_Z13get_global_idj(i32 noundef) #1

declare dso_local spir_func <2 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i(<2 x i4> noundef) #2

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind "uniform-work-group-size"="false" }
attributes #4 = { convergent nounwind willreturn memory(none) }
attributes #5 = { convergent nounwind }
