;; Ensure @llvm.fptosi.sat.* and @llvm.fptoui.sat.* intrinsics are translated

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability Kernel
; CHECK-SPIRV-DAG: Decorate [[SAT1:[0-9]+]] SaturatedConversion
; CHECK-SPIRV-DAG: Decorate [[SAT2:[0-9]+]] SaturatedConversion

; CHECK-SPIRV-DAG: TypeInt [[INT64TY:[0-9]+]] 64
; CHECK-SPIRV-DAG: TypeBool [[BOOLTY:[0-9]+]]
; CHECK-SPIRV-DAG: Constant [[INT64TY]] [[I2SMAX:[0-9]+]] 1
; CHECK-SPIRV-DAG: Constant [[INT64TY]] [[I2SMIN:[0-9]+]] 4294967294 
; CHECK-SPIRV-DAG: ConvertFToS [[INT64TY]] [[SAT1]]
; CHECK-SPIRV-DAG: SGreaterThanEqual [[BOOLTY]] [[SGERES:[0-9]+]] [[SAT1]] [[I2SMAX]]
; CHECK-SPIRV-DAG: SLessThanEqual [[BOOLTY]] [[SLERES:[0-9]+]] [[SAT1]] [[I2SMIN]]
; CHECK-SPIRV-DAG: Select [[INT64TY]] [[SELRES1:[0-9]+]] [[SGERES]] [[I2SMAX]] [[SAT1]]
; CHECK-SPIRV-DAG: Select [[INT64TY]] [[SELRES2:[0-9]+]] [[SLERES]] [[I2SMIN]] [[SELRES1]]

; CHECK-LLVM-DAG: define spir_kernel
; CHECK-LLVM-DAG: %[[R1:[0-9]+]] = {{.*}} i64 {{.*}}convert_long_satf(float %input)
; CHECK-LLVM-DAG: %[[R2:[0-9]+]] = icmp sge i64 %[[R1]], 1
; CHECK-LLVM-DAG: %[[R3:[0-9]+]] = icmp sle i64 %[[R1]], -2
; CHECK-LLVM-DAG: %[[R4:[0-9]+]] = select i1 %[[R2]], i64 1, i64 %[[R1]]
; CHECK-LLVM-DAG: %[[R5:[0-9]+]] = select i1 %[[R3]], i64 -2, i64 %[[R4]]

define spir_kernel void @testfunction_float_to_signed_i2(float %input) {
entry:
   %0 = call i2 @llvm.fptosi.sat.i2.f32(float %input)
   %1 = sext i2 %0 to i64
   ret void
}
declare i2 @llvm.fptosi.sat.i2.f32(float)

; CHECK-SPIRV-DAG: Constant [[INT64TY]] [[I2UMAX:[0-9]+]] 3 
; CHECK-SPIRV-DAG: ConvertFToU [[INT64TY]] [[SAT2]]
; CHECK-SPIRV-DAG: UGreaterThanEqual [[BOOLTY]] [[UGERES:[0-9]+]] [[SAT2]] [[I2UMAX]]
; CHECK-SPIRV-DAG: Select [[INT64TY]] [[SELRES1U:[0-9]+]] [[UGERES]] [[I2UMAX]] [[SAT2]]
; CHECK-LLVM-DAG: define spir_kernel
; CHECK-LLVM-DAG: %[[R1:[0-9]+]] = {{.*}} i64 {{.*}}convert_ulong_satf(float %input)
; CHECK-LLVM-DAG: %[[R2:[0-9]+]] = icmp uge i64 %[[R1]], 3
; CHECK-LLVM-DAG: %[[R3:[0-9]+]] = select i1 %[[R2]], i64 3, i64 %[[R1]]

define spir_kernel void @testfunction_float_to_unsigned_i2(float %input) {
entry:
   %0 = call i2 @llvm.fptoui.sat.i2.f32(float %input)
   %1 = zext i2 %0 to i64
   ret void
}
declare i2 @llvm.fptoui.sat.i2.f32(float)
