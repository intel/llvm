; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -spirv-text --spirv-max-version=1.5 -o %t.txt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llvm-spirv %t.bc -spirv-text --spirv-max-version=1.5 --spirv-ext=+SPV_KHR_integer_dot_product -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv --spirv-max-version=1.5 --spirv-ext=+SPV_KHR_integer_dot_product %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_integer_dot_product -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_integer_dot_product %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc.spvir
; RUN: llvm-dis < %t.rev.bc.spvir | FileCheck %s --check-prefix=CHECK-SPV-IR

; CHECK-ERROR: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_KHR_integer_dot_product

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-NOEXT: 119734787 67072

; CHECK-SPIRV: Int8
; CHECK-SPIRV-DAG: Capability DotProductInput4x8BitKHR
; CHECK-SPIRV-DAG: Capability DotProductInputAllKHR
; CHECK-SPIRV-DAG: Capability DotProductInput4x8BitPackedKHR
; CHECK-SPIRV-DAG: Capability DotProductKHR
; CHECK-SPIRV-EXT: Extension "SPV_KHR_integer_dot_product"
; CHECK-SPIRV-NOEXT-NOT: Extension "SPV_KHR_integer_dot_product"

; CHECK-SPIRV-DAG: TypeInt [[#I8:]] 8
; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32
; CHECK-SPIRV-DAG: TypeInt [[#I64:]] 64

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Packed vector format: 32-bit scalar interpreted as v4i8.
; CHECK-LLVM: @TestNonSatPacked
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i8 @_Z21__spirv_SDotKHR_Rchariii(
; CHECK-LLVM: call spir_func i16 @_Z22__spirv_SDotKHR_Rshortiii(
; CHECK-LLVM: call spir_func i32 @_Z20__spirv_SDotKHR_Rintiii(
; CHECK-LLVM: call spir_func i64 @_Z21__spirv_SDotKHR_Rlongiii(

; CHECK-LLVM: call spir_func i8 @_Z22__spirv_UDotKHR_Ruchariii(
; CHECK-LLVM: call spir_func i16 @_Z23__spirv_UDotKHR_Rushortiii(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_UDotKHR_Ruintiii(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_UDotKHR_Rulongiii(

; CHECK-LLVM: call spir_func i8 @_Z22__spirv_SUDotKHR_Rchariii(
; CHECK-LLVM: call spir_func i16 @_Z23__spirv_SUDotKHR_Rshortiii(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_SUDotKHR_Rintiii(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_SUDotKHR_Rlongiii(

; CHECK-SPV-IR: call spir_func i8 @_Z21__spirv_SDotKHR_Rchariii(
; CHECK-SPV-IR: call spir_func i16 @_Z22__spirv_SDotKHR_Rshortiii(
; CHECK-SPV-IR: call spir_func i32 @_Z20__spirv_SDotKHR_Rintiii(
; CHECK-SPV-IR: call spir_func i64 @_Z21__spirv_SDotKHR_Rlongiii(

; CHECK-SPV-IR: call spir_func i8 @_Z22__spirv_UDotKHR_Rucharjjj(
; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_UDotKHR_Rushortjjj(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_UDotKHR_Ruintjjj(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_UDotKHR_Rulongjjj(

; CHECK-SPV-IR: call spir_func i8 @_Z22__spirv_SUDotKHR_Rchariji(
; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_SUDotKHR_Rshortiji(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_SUDotKHR_Rintiji(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_SUDotKHR_Rlongiji(

; CHECK-SPIRV: 6 SDotKHR [[#I8]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SDotKHR [[#I16]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SDotKHR [[#I32]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SDotKHR [[#I64]] [[#]] [[#]] [[#]] 0

; CHECK-SPIRV: 6 UDotKHR [[#I8]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 UDotKHR [[#I16]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 UDotKHR [[#I32]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 UDotKHR [[#I64]] [[#]] [[#]] [[#]] 0

; CHECK-SPIRV: 6 SUDotKHR [[#I8]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SUDotKHR [[#I16]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SUDotKHR [[#I32]] [[#]] [[#]] [[#]] 0
; CHECK-SPIRV: 6 SUDotKHR [[#I64]] [[#]] [[#]] [[#]] 0

; Function Attrs: nounwind
define spir_kernel void @TestNonSatPacked(i32 %0, i32 %1) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
  %3 = call spir_func i8 @_Z21__spirv_SDotKHR_Rchariii(i32 %0, i32 %1, i32 0) #0
  %4 = call spir_func i16 @_Z22__spirv_SDotKHR_Rshortiii(i32 %0, i32 %1, i32 0) #0
  %5 = call spir_func i32 @_Z20__spirv_SDotKHR_Rintiii(i32 %0, i32 %1, i32 0) #0
  %6 = call spir_func i64 @_Z21__spirv_SDotKHR_Rlongiii(i32 %0, i32 %1, i32 0) #0

  %7 = call spir_func i8 @_Z22__spirv_UDotKHR_Ruchariii(i32 %0, i32 %1, i32 0) #0
  %8 = call spir_func i16 @_Z23__spirv_UDotKHR_Rushortiii(i32 %0, i32 %1, i32 0) #0
  %9 = call spir_func i32 @_Z21__spirv_UDotKHR_Ruintiii(i32 %0, i32 %1, i32 0) #0
  %10 = call spir_func i64 @_Z22__spirv_UDotKHR_Rulongiii(i32 %0, i32 %1, i32 0) #0

  %11 = call spir_func i8 @_Z22__spirv_SUDotKHR_Rchariii(i32 %0, i32 %1, i32 0) #0
  %12 = call spir_func i16 @_Z23__spirv_SUDotKHR_Rshortiii(i32 %0, i32 %1, i32 0) #0
  %13 = call spir_func i32 @_Z21__spirv_SUDotKHR_Rintiii(i32 %0, i32 %1, i32 0) #0
  %14 = call spir_func i64 @_Z22__spirv_SUDotKHR_Rlongiii(i32 %0, i32 %1, i32 0) #0
  ret void
}

; Vector format: v4i8.
; CHECK-LLVM: @TestNonSatVec
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i8 @_Z21__spirv_SDotKHR_RcharDv4_cS_(
; CHECK-LLVM: call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv4_cS_(
; CHECK-LLVM: call spir_func i32 @_Z20__spirv_SDotKHR_RintDv4_cS_(
; CHECK-LLVM: call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv4_cS_(

; CHECK-LLVM: call spir_func i8 @_Z22__spirv_UDotKHR_RucharDv4_cS_(
; CHECK-LLVM: call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv4_cS_(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv4_cS_(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv4_cS_(

; CHECK-LLVM: call spir_func i8 @_Z22__spirv_SUDotKHR_RcharDv4_cS_(
; CHECK-LLVM: call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv4_cS_(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv4_cS_(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv4_cS_(

; CHECK-SPV-IR: call spir_func i8 @_Z21__spirv_SDotKHR_RcharDv4_cS_(
; CHECK-SPV-IR: call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv4_cS_(
; CHECK-SPV-IR: call spir_func i32 @_Z20__spirv_SDotKHR_RintDv4_cS_(
; CHECK-SPV-IR: call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv4_cS_(

; CHECK-SPV-IR: call spir_func i8 @_Z22__spirv_UDotKHR_RucharDv4_hS_(
; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv4_hS_(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv4_hS_(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv4_hS_(

; CHECK-SPV-IR: call spir_func i8 @_Z22__spirv_SUDotKHR_RcharDv4_cDv4_h(
; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv4_cDv4_h(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv4_cDv4_h(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv4_cDv4_h(

; CHECK-SPIRV: 5 SDotKHR [[#I8]]
; CHECK-SPIRV: 5 SDotKHR [[#I16]]
; CHECK-SPIRV: 5 SDotKHR [[#I32]]
; CHECK-SPIRV: 5 SDotKHR [[#I64]]

; CHECK-SPIRV: 5 UDotKHR [[#I8]]
; CHECK-SPIRV: 5 UDotKHR [[#I16]]
; CHECK-SPIRV: 5 UDotKHR [[#I32]]
; CHECK-SPIRV: 5 UDotKHR [[#I64]]

; CHECK-SPIRV: 5 SUDotKHR [[#I8]]
; CHECK-SPIRV: 5 SUDotKHR [[#I16]]
; CHECK-SPIRV: 5 SUDotKHR [[#I32]]
; CHECK-SPIRV: 5 SUDotKHR [[#I64]]

; Function Attrs: nounwind
define spir_kernel void @TestNonSatVec(<4 x i8> %0, <4 x i8> %1) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !9 !kernel_arg_type_qual !8 !kernel_arg_base_type !9 {
  %3 = call spir_func i8 @_Z21__spirv_SDotKHR_RcharDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %4 = call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %5 = call spir_func i32 @_Z20__spirv_SDotKHR_RintDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %6 = call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0

  %7 = call spir_func i8 @_Z22__spirv_UDotKHR_RucharDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %8 = call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %9 = call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %10 = call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0

  %11 = call spir_func i8 @_Z22__spirv_SUDotKHR_RcharDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %12 = call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %13 = call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  %14 = call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv4_cS_(<4 x i8> %0, <4 x i8> %1) #0
  ret void
}

; Vector format: v2i16, which is a case of DotProductInputAllKHR.
; CHECK-LLVM: @TestNonSatAll
; CHECK-SPIRV: Function

; CHECK-LLVM: call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv2_sS_(
; CHECK-LLVM: call spir_func i32 @_Z20__spirv_SDotKHR_RintDv2_sS_(
; CHECK-LLVM: call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv2_sS_(

; CHECK-LLVM: call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv2_sS_(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv2_sS_(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv2_sS_(

; CHECK-LLVM: call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv2_sS_(
; CHECK-LLVM: call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv2_sS_(
; CHECK-LLVM: call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv2_sS_(

; CHECK-SPV-IR: call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv2_sS_(
; CHECK-SPV-IR: call spir_func i32 @_Z20__spirv_SDotKHR_RintDv2_sS_(
; CHECK-SPV-IR: call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv2_sS_(

; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv2_tS_(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv2_tS_(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv2_tS_(

; CHECK-SPV-IR: call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv2_sDv2_t(
; CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv2_sDv2_t(
; CHECK-SPV-IR: call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv2_sDv2_t(

; CHECK-SPIRV: 5 SDotKHR [[#I16]]
; CHECK-SPIRV: 5 SDotKHR [[#I32]]
; CHECK-SPIRV: 5 SDotKHR [[#I64]]

; CHECK-SPIRV: 5 UDotKHR [[#I16]]
; CHECK-SPIRV: 5 UDotKHR [[#I32]]
; CHECK-SPIRV: 5 UDotKHR [[#I64]]

; CHECK-SPIRV: 5 SUDotKHR [[#I16]]
; CHECK-SPIRV: 5 SUDotKHR [[#I32]]
; CHECK-SPIRV: 5 SUDotKHR [[#I64]]

; Function Attrs: nounwind
define spir_kernel void @TestNonSatAll(<2 x i16> %0, <2 x i16> %1) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !10 !kernel_arg_type_qual !8 !kernel_arg_base_type !10 {
  %3 = call spir_func i16 @_Z22__spirv_SDotKHR_RshortDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %4 = call spir_func i32 @_Z20__spirv_SDotKHR_RintDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %5 = call spir_func i64 @_Z21__spirv_SDotKHR_RlongDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0

  %6 = call spir_func i16 @_Z23__spirv_UDotKHR_RushortDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %7 = call spir_func i32 @_Z21__spirv_UDotKHR_RuintDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %8 = call spir_func i64 @_Z22__spirv_UDotKHR_RulongDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0

  %9 = call spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %10 = call spir_func i32 @_Z21__spirv_SUDotKHR_RintDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  %11 = call spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv2_sS_(<2 x i16> %0, <2 x i16> %1) #0
  ret void
}

; Function Attrs: nounwind
declare spir_func i8 @_Z21__spirv_SDotKHR_Rchariii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z22__spirv_SDotKHR_Rshortiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z20__spirv_SDotKHR_Rintiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z21__spirv_SDotKHR_Rlongiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i8 @_Z22__spirv_UDotKHR_Ruchariii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_UDotKHR_Rushortiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_UDotKHR_Ruintiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_UDotKHR_Rulongiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i8 @_Z22__spirv_SUDotKHR_Rchariii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_SUDotKHR_Rshortiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_SUDotKHR_Rintiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_SUDotKHR_Rlongiii(i32, i32, i32) #0

; Function Attrs: nounwind
declare spir_func i8 @_Z21__spirv_SDotKHR_RcharDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z22__spirv_SDotKHR_RshortDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z20__spirv_SDotKHR_RintDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z21__spirv_SDotKHR_RlongDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i8 @_Z22__spirv_UDotKHR_RucharDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_UDotKHR_RushortDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_UDotKHR_RuintDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_UDotKHR_RulongDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i8 @_Z22__spirv_SUDotKHR_RcharDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_SUDotKHR_RintDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv4_cS_(<4 x i8>, <4 x i8>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z22__spirv_SDotKHR_RshortDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z20__spirv_SDotKHR_RintDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z21__spirv_SDotKHR_RlongDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_UDotKHR_RushortDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_UDotKHR_RuintDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_UDotKHR_RulongDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i16 @_Z23__spirv_SUDotKHR_RshortDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z21__spirv_SUDotKHR_RintDv2_sS_(<2 x i16>, <2 x i16>) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z22__spirv_SUDotKHR_RlongDv2_sS_(<2 x i16>, <2 x i16>) #0

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
!6 = !{!"none", !"none"}
!7 = !{!"int", !"int"}
!8 = !{!"", !""}
!9 = !{!"char4", !"char4"}
!10 = !{!"short2", !"short2"}
