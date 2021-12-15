; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -spirv-text -o %t.txt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_integer_dot_product -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_integer_dot_product %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-ERROR: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_KHR_integer_dot_product

; CHECK-SPIRV: Capability DotProductInputAllKHR
; CHECK-SPIRV: Capability DotProductInput4x8BitKHR
; CHECK-SPIRV: Capability DotProductInput4x8BitPackedKHR
; CHECK-SPIRV: Capability DotProductKHR
; CHECK-SPIRV: Extension "SPV_KHR_integer_dot_product"

; CHECK-SPIRV-DAG: TypeInt [[#I8:]] 8
; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32
; CHECK-SPIRV-DAG: TypeInt [[#I64:]] 64

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Packed vector format: 32-bit scalar interpreted as v4i8.
; CHECK-LLVM: @TestSatPacked
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i8 @_Z27__spirv_SDotAccSatKHR_Rchariici
; CHECK-LLVM: call spir_func i16 @_Z28__spirv_SDotAccSatKHR_Rshortiisi
; CHECK-LLVM: call spir_func i32 @_Z26__spirv_SDotAccSatKHR_Rintiiii
; CHECK-LLVM: call spir_func i64 @_Z27__spirv_SDotAccSatKHR_Rlongiili

; CHECK-LLVM: call spir_func i8 @_Z28__spirv_UDotAccSatKHR_Ruchariici
; CHECK-LLVM: call spir_func i16 @_Z29__spirv_UDotAccSatKHR_Rushortiisi
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_UDotAccSatKHR_Ruintiiii
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_UDotAccSatKHR_Rulongiili

; CHECK-LLVM: call spir_func i8 @_Z28__spirv_SUDotAccSatKHR_Rchariici
; CHECK-LLVM: call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_Rshortiisi
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_Rintiiii
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_Rlongiili

; CHECK-SPIRV: 7 SDotAccSatKHR [[#I8]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SDotAccSatKHR [[#I16]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SDotAccSatKHR [[#I32]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SDotAccSatKHR [[#I64]] [[#]] [[#]] [[#]] [[#]] 0

; CHECK-SPIRV: 7 UDotAccSatKHR [[#I8]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 UDotAccSatKHR [[#I16]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 UDotAccSatKHR [[#I32]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 UDotAccSatKHR [[#I64]] [[#]] [[#]] [[#]] [[#]] 0

; CHECK-SPIRV: 7 SUDotAccSatKHR [[#I8]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SUDotAccSatKHR [[#I16]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SUDotAccSatKHR [[#I32]] [[#]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 7 SUDotAccSatKHR [[#I64]] [[#]] [[#]] [[#]] [[#]] 0

; Function Attrs: nounwind
define spir_kernel void @TestSatPacked(i32 %0, i32 %1, i8 %acc8, i16 %acc16, i32 %acc32, i64 %acc64) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
  %3 = call spir_func i8 @_Z27__spirv_SDotAccSatKHR_Rchariici(i32 %0, i32 %1, i8 %acc8, i32 0) #0
  %4 = call spir_func i16 @_Z28__spirv_SDotAccSatKHR_Rshortiisi(i32 %0, i32 %1, i16 %acc16, i32 0) #0
  %5 = call spir_func i32 @_Z26__spirv_SDotAccSatKHR_Rintiiii(i32 %0, i32 %1, i32 %acc32, i32 0) #0
  %6 = call spir_func i64 @_Z27__spirv_SDotAccSatKHR_Rlongiili(i32 %0, i32 %1, i64 %acc64, i32 0) #0

  %7 = call spir_func i8 @_Z28__spirv_UDotAccSatKHR_Ruchariici(i32 %0, i32 %1, i8 %acc8, i32 0) #0
  %8 = call spir_func i16 @_Z29__spirv_UDotAccSatKHR_Rushortiisi(i32 %0, i32 %1, i16 %acc16, i32 0) #0
  %9 = call spir_func i32 @_Z27__spirv_UDotAccSatKHR_Ruintiiii(i32 %0, i32 %1, i32 %acc32, i32 0) #0
  %10 = call spir_func i64 @_Z28__spirv_UDotAccSatKHR_Rulongiili(i32 %0, i32 %1, i64 %acc64, i32 0) #0

  %11 = call spir_func i8 @_Z28__spirv_SUDotAccSatKHR_Rchariici(i32 %0, i32 %1, i8 %acc8, i32 0) #0
  %12 = call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_Rshortiisi(i32 %0, i32 %1, i16 %acc16, i32 0) #0
  %13 = call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_Rintiiii(i32 %0, i32 %1, i32 %acc32, i32 0) #0
  %14 = call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_Rlongiili(i32 %0, i32 %1, i64 %acc64, i32 0) #0
  ret void
}

; Vector format: v4i8.
; CHECK-LLVM: @TestSatVec
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i8 @_Z27__spirv_SDotAccSatKHR_RcharDv4_cS_
; CHECK-LLVM: call spir_func i16 @_Z28__spirv_SDotAccSatKHR_RshortDv4_cS_
; CHECK-LLVM: call spir_func i32 @_Z26__spirv_SDotAccSatKHR_RintDv4_cS_
; CHECK-LLVM: call spir_func i64 @_Z27__spirv_SDotAccSatKHR_RlongDv4_cS_

; CHECK-LLVM: call spir_func i8 @_Z28__spirv_UDotAccSatKHR_RucharDv4_cS_
; CHECK-LLVM: call spir_func i16 @_Z29__spirv_UDotAccSatKHR_RushortDv4_cS_
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_UDotAccSatKHR_RuintDv4_cS_
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_UDotAccSatKHR_RulongDv4_cS_

; CHECK-LLVM: call spir_func i8 @_Z28__spirv_SUDotAccSatKHR_RcharDv4_cS_
; CHECK-LLVM: call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv4_cS_
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_RintDv4_cS_
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv4_cS_

; CHECK-SPIRV: 6 SDotAccSatKHR [[#I8]]
; CHECK-SPIRV: 6 SDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 SDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 SDotAccSatKHR [[#I64]]

; CHECK-SPIRV: 6 UDotAccSatKHR [[#I8]]
; CHECK-SPIRV: 6 UDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 UDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 UDotAccSatKHR [[#I64]]

; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I8]]
; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I64]]

; Function Attrs: nounwind
define spir_kernel void @TestSatVec(<4 x i8> %0, <4 x i8> %1, i8 %acc8, i16 %acc16, i32 %acc32, i64 %acc64) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !9 !kernel_arg_type_qual !8 !kernel_arg_base_type !9 {
  %3 = call spir_func i8 @_Z27__spirv_SDotAccSatKHR_RcharDv4_cS_c(<4 x i8> %0, <4 x i8> %1, i8 %acc8) #0
  %4 = call spir_func i16 @_Z28__spirv_SDotAccSatKHR_RshortDv4_cS_s(<4 x i8> %0, <4 x i8> %1, i16 %acc16) #0
  %5 = call spir_func i32 @_Z26__spirv_SDotAccSatKHR_RintDv4_cS_i(<4 x i8> %0, <4 x i8> %1, i32 %acc32) #0
  %6 = call spir_func i64 @_Z27__spirv_SDotAccSatKHR_RlongDv4_cS_l(<4 x i8> %0, <4 x i8> %1, i64 %acc64) #0

  %7 = call spir_func i8 @_Z28__spirv_UDotAccSatKHR_RucharDv4_cS_c(<4 x i8> %0, <4 x i8> %1, i8 %acc8) #0
  %8 = call spir_func i16 @_Z29__spirv_UDotAccSatKHR_RushortDv4_cS_s(<4 x i8> %0, <4 x i8> %1, i16 %acc16) #0
  %9 = call spir_func i32 @_Z27__spirv_UDotAccSatKHR_RuintDv4_cS_i(<4 x i8> %0, <4 x i8> %1, i32 %acc32) #0
  %10 = call spir_func i64 @_Z28__spirv_UDotAccSatKHR_RulongDv4_cS_l(<4 x i8> %0, <4 x i8> %1, i64 %acc64) #0

  %11 = call spir_func i8 @_Z28__spirv_SUDotAccSatKHR_RcharDv4_cS_c(<4 x i8> %0, <4 x i8> %1, i8 %acc8) #0
  %12 = call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv4_cS_s(<4 x i8> %0, <4 x i8> %1, i16 %acc16) #0
  %13 = call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_RintDv4_cS_i(<4 x i8> %0, <4 x i8> %1, i32 %acc32) #0
  %14 = call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv4_cS_l(<4 x i8> %0, <4 x i8> %1, i64 %acc64) #0
  ret void
}

; Vector format: v2i16, which is a case of DotProductInputAllKHR.
; CHECK-LLVM: @TestSatAll
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i16 @_Z28__spirv_SDotAccSatKHR_RshortDv2_sS_
; CHECK-LLVM: call spir_func i32 @_Z26__spirv_SDotAccSatKHR_RintDv2_sS_
; CHECK-LLVM: call spir_func i64 @_Z27__spirv_SDotAccSatKHR_RlongDv2_sS_

; CHECK-LLVM: call spir_func i16 @_Z29__spirv_UDotAccSatKHR_RushortDv2_sS_
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_UDotAccSatKHR_RuintDv2_sS_
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_UDotAccSatKHR_RulongDv2_sS_

; CHECK-LLVM: call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv2_sS_
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_RintDv2_sS_
; CHECK-LLVM: call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv2_sS_

; CHECK-SPIRV: 6 SDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 SDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 SDotAccSatKHR [[#I64]]

; CHECK-SPIRV: 6 UDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 UDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 UDotAccSatKHR [[#I64]]

; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I16]]
; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I32]]
; CHECK-SPIRV: 6 SUDotAccSatKHR [[#I64]]

; Function Attrs: nounwind
define spir_kernel void @TestSatAll(<2 x i16> %0, <2 x i16> %1, i8 %acc8, i16 %acc16, i32 %acc32, i64 %acc64) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !10 !kernel_arg_type_qual !8 !kernel_arg_base_type !10 {
  %3 = call spir_func i16 @_Z28__spirv_SDotAccSatKHR_RshortDv2_sS_s(<2 x i16> %0, <2 x i16> %1, i16 %acc16) #0
  %4 = call spir_func i32 @_Z26__spirv_SDotAccSatKHR_RintDv2_sS_i(<2 x i16> %0, <2 x i16> %1, i32 %acc32) #0
  %5 = call spir_func i64 @_Z27__spirv_SDotAccSatKHR_RlongDv2_sS_l(<2 x i16> %0, <2 x i16> %1, i64 %acc64) #0

  %6 = call spir_func i16 @_Z29__spirv_UDotAccSatKHR_RushortDv2_sS_s(<2 x i16> %0, <2 x i16> %1, i16 %acc16) #0
  %7 = call spir_func i32 @_Z27__spirv_UDotAccSatKHR_RuintDv2_sS_i(<2 x i16> %0, <2 x i16> %1, i32 %acc32) #0
  %8 = call spir_func i64 @_Z28__spirv_UDotAccSatKHR_RulongDv2_sS_l(<2 x i16> %0, <2 x i16> %1, i64 %acc64) #0

  %9 = call spir_func i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv2_sS_s(<2 x i16> %0, <2 x i16> %1, i16 %acc16) #0
  %10 = call spir_func i32 @_Z27__spirv_SUDotAccSatKHR_RintDv2_sS_i(<2 x i16> %0, <2 x i16> %1, i32 %acc32) #0
  %11 = call spir_func i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv2_sS_l(<2 x i16> %0, <2 x i16> %1, i64 %acc64) #0
  ret void
}

; Function Attrs: alwaysinline nounwind
declare i8 @_Z27__spirv_SDotAccSatKHR_Rchariici(i32, i32, i8, i32) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z28__spirv_SDotAccSatKHR_Rshortiisi(i32, i32, i16, i32) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z26__spirv_SDotAccSatKHR_Rintiiii(i32, i32, i32, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z27__spirv_SDotAccSatKHR_Rlongiili(i32, i32, i64, i32) #0

; Function Attrs: alwaysinline nounwind
declare i8 @_Z28__spirv_UDotAccSatKHR_Ruchariici(i32, i32, i8, i32) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z29__spirv_UDotAccSatKHR_Rushortiisi(i32, i32, i16, i32) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z27__spirv_UDotAccSatKHR_Ruintiiii(i32, i32, i32, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z28__spirv_UDotAccSatKHR_Rulongiili(i32, i32, i64, i32) #0

; Function Attrs: alwaysinline nounwind
declare i8 @_Z28__spirv_SUDotAccSatKHR_Rchariici(i32, i32, i8, i32) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z29__spirv_SUDotAccSatKHR_Rshortiisi(i32, i32, i16, i32) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z27__spirv_SUDotAccSatKHR_Rintiiii(i32, i32, i32, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z28__spirv_SUDotAccSatKHR_Rlongiili(i32, i32, i64, i32) #0

; Function Attrs: alwaysinline nounwind
declare i8 @_Z27__spirv_SDotAccSatKHR_RcharDv4_cS_c(<4 x i8>, <4 x i8>, i8) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z28__spirv_SDotAccSatKHR_RshortDv4_cS_s(<4 x i8>, <4 x i8>, i16) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z26__spirv_SDotAccSatKHR_RintDv4_cS_i(<4 x i8>, <4 x i8>, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z27__spirv_SDotAccSatKHR_RlongDv4_cS_l(<4 x i8>, <4 x i8>, i64) #0

; Function Attrs: alwaysinline nounwind
declare i8 @_Z28__spirv_UDotAccSatKHR_RucharDv4_cS_c(<4 x i8>, <4 x i8>, i8) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z29__spirv_UDotAccSatKHR_RushortDv4_cS_s(<4 x i8>, <4 x i8>, i16) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z27__spirv_UDotAccSatKHR_RuintDv4_cS_i(<4 x i8>, <4 x i8>, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z28__spirv_UDotAccSatKHR_RulongDv4_cS_l(<4 x i8>, <4 x i8>, i64) #0

; Function Attrs: alwaysinline nounwind
declare i8 @_Z28__spirv_SUDotAccSatKHR_RcharDv4_cS_c(<4 x i8>, <4 x i8>, i8) #0

; Function Attrs: alwaysinline nounwind
declare i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv4_cS_s(<4 x i8>, <4 x i8>, i16) #0

; Function Attrs: alwaysinline nounwind
declare i32 @_Z27__spirv_SUDotAccSatKHR_RintDv4_cS_i(<4 x i8>, <4 x i8>, i32) #0

; Function Attrs: alwaysinline nounwind
declare i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv4_cS_l(<4 x i8>, <4 x i8>, i64) #0

; Function Attrs: nounwind
declare i16 @_Z28__spirv_SDotAccSatKHR_RshortDv2_sS_s(<2 x i16>, <2 x i16>, i16) #0

; Function Attrs: nounwind
declare i32 @_Z26__spirv_SDotAccSatKHR_RintDv2_sS_i(<2 x i16>, <2 x i16>, i32) #0

; Function Attrs: nounwind
declare i64 @_Z27__spirv_SDotAccSatKHR_RlongDv2_sS_l(<2 x i16>, <2 x i16>, i64) #0

; Function Attrs: nounwind
declare i16 @_Z29__spirv_UDotAccSatKHR_RushortDv2_sS_s(<2 x i16>, <2 x i16>, i16) #0

; Function Attrs: nounwind
declare i32 @_Z27__spirv_UDotAccSatKHR_RuintDv2_sS_i(<2 x i16>, <2 x i16>, i32) #0

; Function Attrs: nounwind
declare i64 @_Z28__spirv_UDotAccSatKHR_RulongDv2_sS_l(<2 x i16>, <2 x i16>, i64) #0

; Function Attrs: nounwind
declare i16 @_Z29__spirv_SUDotAccSatKHR_RshortDv2_sS_s(<2 x i16>, <2 x i16>, i16) #0

; Function Attrs: nounwind
declare i32 @_Z27__spirv_SUDotAccSatKHR_RintDv2_sS_i(<2 x i16>, <2 x i16>, i32) #0

; Function Attrs: nounwind
declare i64 @_Z28__spirv_SUDotAccSatKHR_RlongDv2_sS_l(<2 x i16>, <2 x i16>, i64) #0

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 7, i16 0}
!5 = !{i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!7 = !{!"int", !"int", !"char", !"short", !"int", !"long"}
!8 = !{!"", !"", !"", !"", !"", !"", !""}
!9 = !{!"char4", !"char4", !"char", !"short", !"int", !"long"}
!10 = !{!"short2", !"short2", !"char", !"short", !"int", !"long"}
