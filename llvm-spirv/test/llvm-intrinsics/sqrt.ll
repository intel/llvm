; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

; CHECK: ExtInstImport [[ExtInstSetId:[0-9]+]] "OpenCL.std"
; CHECK: TypeFloat [[Float:[0-9]+]] 32
; CHECK: TypeFloat [[Double:[0-9]+]] 64
; CHECK: TypeVector [[Double4:[0-9]+]] [[Double]] 4
; CHECK: Constant [[Float]] [[FloatArg:[0-9]+]]
; CHECK: Constant [[Double]] [[DoubleArg:[0-9]+]]
; CHECK: ConstantComposite [[Double4]] [[Double4Arg:[0-9]+]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: noinline nounwind optnone
define spir_func void @test_sqrt() #0 {
entry:
  %0 = call float @llvm.sqrt.f32(float 0x40091EB860000000)
  %1 = call double @llvm.sqrt.f64(double 2.710000e+00)
  %2 = call <4 x double> @llvm.sqrt.v4f64(<4 x double> <double 5.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01>)
; CHECK: ExtInst [[Float]] {{[0-9]+}} [[ExtInstSetId]] sqrt [[FloatArg]]
; CHECK: ExtInst [[Double]] {{[0-9]+}} [[ExtInstSetId]] sqrt [[DoubleArg]]
; CHECK: ExtInst [[Double4]] {{[0-9]+}} [[ExtInstSetId]] sqrt [[Double4Arg]]
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git b89131cdda5871731a9139664aef2b70c6d72bbd)"}

