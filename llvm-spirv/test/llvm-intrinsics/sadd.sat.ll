; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_func i32 @Test(i32 %lhs, i32 %rhs) local_unnamed_addr #0 {
entry:
  %0 = call i32 @llvm.sadd.sat.i32(i32 %lhs, i32 %rhs)
  ret i32 %0
}

; CHECK-DAG: Constant {{[0-9]+}} [[int_min:[0-9]+]] 2147483648
; CHECK-DAG: Constant {{[0-9]+}} [[c31:[0-9]+]] 31

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[lhs:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[rhs:[0-9]+]]
; CHECK: FunctionCall {{[0-9]+}} [[struct_res:[0-9]+]] {{[0-9]+}} [[lhs]] [[rhs]]
; CHECK: CompositeExtract {{[0-9]+}} [[sum:[0-9]+]] [[struct_res]] 0
; CHECK: CompositeExtract {{[0-9]+}} [[overflow:[0-9]+]] [[struct_res]] 1
; CHECK: ShiftRightArithmetic {{[0-9]+}} [[shift:[0-9]+]] [[sum]] [[c31]]
; CHECK: BitwiseXor {{[0-9]+}} [[overflowed:[0-9]+]] [[shift]] [[int_min]]
; CHECK: Select {{[0-9]+}} [[result:[0-9]+]] [[overflow]] [[overflowed]] [[sum]]
; CHECK: ReturnValue [[result]]

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.sadd.sat.i32(i32, i32) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
