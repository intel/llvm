; REQUIRES: spirv-dis
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-dis --raw-id %t.spv | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: spirv-val %t.spv 

; CHECK-SPIRV:          [[opencl:%[0-9]+]] = OpExtInstImport "OpenCL.std"
; CHECK-SPIRV-DAG:      [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG:      [[i64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG:      [[f32:%[0-9]+]] = OpTypeFloat 32
; CHECK-SPIRV-DAG:      [[f64:%[0-9]+]] = OpTypeFloat 64
; CHECK-SPIRV:      [[rounded_f32:%[0-9]+]] = OpExtInst [[f32]] [[opencl]] round
; CHECK-SPIRV:                        OpConvertFToS [[i32]] [[rounded_f32]]
; CHECK-SPIRV:      [[rounded_f64:%[0-9]+]] = OpExtInst [[f64]] [[opencl]] round
; CHECK-SPIRV:                        OpConvertFToS [[i64]] [[rounded_f64]]

target triple = "spir64-unknown-unknown"

define spir_func i32 @test_0(float %arg0) {
entry:
  %0 = call i32 @llvm.lround.i32.f32(float %arg0)
  ret i32 %0
}

define spir_func i64 @test_1(double %arg0) {
entry:
  %0 = call i64 @llvm.lround.i64.f64(double %arg0)
  ret i64 %0
}

declare i32 @llvm.lround.i32.f32(float)
declare i64 @llvm.lround.i64.f64(double)
