; RUN: llvm-as %s -o %t.bc

; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-WO-EXT

; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=+SPV_KHR_uniform_group_instructions
; RUN: llvm-spirv %t.spv -o %t.spt -to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv --spirv-target-env=SPV-IR -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-SPIRV

; CHECK-WO-EXT: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-WO-EXT: SPV_KHR_uniform_group_instructions

; CHECK-SPIRV: Capability GroupUniformArithmeticKHR
; CHECK-SPIRV: Extension "SPV_KHR_uniform_group_instructions"
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Scope:]] 2
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Val1:]] 0
; CHECK-SPIRV: TypeBool [[#TypeBool:]]
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 16
; CHECK-SPIRV: ConstantFalse [[#TypeBool]] [[#ConstFalse:]]
; CHECK-SPIRV: Constant [[#TypeFloat]] [[#Val2:]]

; CHECK-SPIRV: GroupBitwiseAndKHR [[#TypeInt]] [[#]] [[#Scope]] 0 [[#Val1]]
; CHECK-SPIRV: GroupBitwiseOrKHR [[#TypeInt]] [[#]] [[#Scope]] 0 [[#Val1]]
; CHECK-SPIRV: GroupBitwiseXorKHR [[#TypeInt]] [[#]] [[#Scope]] 0 [[#Val1]]
; CHECK-SPIRV: GroupLogicalAndKHR [[#TypeBool]] [[#]] [[#Scope]] 0 [[#ConstFalse]]
; CHECK-SPIRV: GroupLogicalOrKHR [[#TypeBool]] [[#]] [[#Scope]] 0 [[#ConstFalse]]
; CHECK-SPIRV: GroupLogicalXorKHR [[#TypeBool]] [[#]] [[#Scope]] 0 [[#ConstFalse]]
; CHECK-SPIRV: GroupIMulKHR [[#TypeInt]] [[#]] [[#Scope]] 0 [[#Val1]]
; CHECK-SPIRV: GroupFMulKHR [[#TypeFloat]] [[#]] [[#Scope]] 0 [[#Val2]]

; CHECK-LLVM: call spir_func i32 @_Z29work_group_reduce_bitwise_andi(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z28work_group_reduce_bitwise_ori(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z29work_group_reduce_bitwise_xori(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z29work_group_reduce_logical_andi(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z28work_group_reduce_logical_ori(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z29work_group_reduce_logical_xori(i32 0)
; CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_muli(i32 0)
; CHECK-LLVM: call spir_func half @_Z21work_group_reduce_mulDh(half 0xH0000)

; CHECK-LLVM-SPIRV: %call1 = call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHR{{.*}}(i32 2, i32 0, i32 0)
; CHECK-LLVM-SPIRV: %call2 = call spir_func i32 @_Z25__spirv_GroupBitwiseOrKHR{{.*}}(i32 2, i32 0, i32 0)
; CHECK-LLVM-SPIRV: %call3 = call spir_func i32 @_Z26__spirv_GroupBitwiseXorKHR{{.*}}(i32 2, i32 0, i32 0)
; CHECK-LLVM-SPIRV: %call4 = call spir_func i1 @_Z26__spirv_GroupLogicalAndKHR{{.*}}(i32 2, i32 0, i1 false)
; CHECK-LLVM-SPIRV: %call5 = call spir_func i1 @_Z25__spirv_GroupLogicalOrKHR{{.*}}(i32 2, i32 0, i1 false)
; CHECK-LLVM-SPIRV: %call6 = call spir_func i1 @_Z26__spirv_GroupLogicalXorKHR{{.*}}(i32 2, i32 0, i1 false)
; CHECK-LLVM-SPIRV: %call7 = call spir_func i32 @_Z20__spirv_GroupIMulKHR{{.*}}(i32 2, i32 0, i32 0)
; CHECK-LLVM-SPIRV: %call8 = call spir_func half @_Z20__spirv_GroupFMulKHR{{.*}}(i32 2, i32 0, half 0xH0000)

; ModuleID = 'source.bc'
source_filename = "group_operations.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse
define dso_local spir_func void @_Z10test_groupN2cl4sycl5groupILi1EEE() local_unnamed_addr #0 {
entry:
  %call1 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHRjji(i32 2, i32 0, i32 0) #2
  %call2 = tail call spir_func i32 @_Z25__spirv_GroupBitwiseOrKHRjji(i32 2, i32 0, i32 0) #2
  %call3 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseXorKHRjji(i32 2, i32 0, i32 0) #2
  %call4 = tail call spir_func i1 @_Z26__spirv_GroupLogicalAndKHRjji(i32 2, i32 0, i1 false) #2
  %call5 = tail call spir_func i1 @_Z25__spirv_GroupLogicalOrKHRjji(i32 2, i32 0, i1 false) #2
  %call6 = tail call spir_func i1 @_Z26__spirv_GroupLogicalXorKHRjji(i32 2, i32 0, i1 false) #2
  %call7 = tail call spir_func i32 @_Z20__spirv_GroupIMulKHRjji(i32 2, i32 0, i32 0) #2
  %call8 = tail call fast spir_func half @_Z20__spirv_GroupFMulKHRjjDF16_(i32 2, i32 0, half 0xH0000) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z26__spirv_GroupBitwiseAndKHRjji(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25__spirv_GroupBitwiseOrKHRjji(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z26__spirv_GroupBitwiseXorKHRjji(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i1 @_Z26__spirv_GroupLogicalAndKHRjji(i32, i32, i1) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i1 @_Z25__spirv_GroupLogicalOrKHRjji(i32, i32, i1) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i1 @_Z26__spirv_GroupLogicalXorKHRjji(i32, i32, i1) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z20__spirv_GroupIMulKHRjji(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func half @_Z20__spirv_GroupFMulKHRjjDF16_(i32, i32, half) local_unnamed_addr #1

attributes #0 = { convergent norecurse "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="all" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="group_operations.cpp" "unsafe-fp-math"="true" }
attributes #1 = { convergent "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { convergent }
