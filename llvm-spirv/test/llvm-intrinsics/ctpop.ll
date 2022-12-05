; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv


; CHECK: BitCount {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK: BitCount {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK: BitCount {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK: BitCount {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK: BitCount {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

; Function Attrs: norecurse nounwind readnone
define dso_local spir_kernel void @test(i8 %x8, i16 %x16, i32 %x32, i64 %x64, <2 x i32> %x2i32) local_unnamed_addr #0 !kernel_arg_buffer_location !5 {
entry:
  %0 = tail call i8 @llvm.ctpop.i8(i8 %x8) #2
  %1 = tail call i16 @llvm.ctpop.i16(i16 %x16) #2
  %2 = tail call i32 @llvm.ctpop.i32(i32 %x32) #2
  %3 = tail call i64 @llvm.ctpop.i64(i64 %x64) #2
  %4 = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x2i32) #2
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i8 @llvm.ctpop.i8(i8  ) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i16 @llvm.ctpop.i16(i16 ) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i32 @llvm.ctpop.i32(i32 ) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i64 @llvm.ctpop.i64(i64 ) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32> ) #1

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cl" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2, !2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 12.0.0 (https://github.com/c199914007/llvm.git 0051629b5f4d81af1b049da17bce0b00f03998f8)"}
!5 = !{i32 -1}
