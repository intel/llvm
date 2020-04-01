; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv

; CHECK-SPIRV: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-SPIRV: Constant [[Int]] [[MemScope_Device:[0-9]+]] 1
; CHECK-SPIRV: Constant [[Int]] [[MemSemEqual_SeqCst:[0-9]+]] 16
; CHECK-SPIRV: Constant [[Int]] [[MemSemUnequal_Acquire:[0-9]+]] 2

; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[Pointer:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[Value_ptr:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[Comparator:[0-9]+]]

; CHECK-SPIRV: Load [[Int]] [[Value:[0-9]+]] [[Value_ptr]]
; CHECK-SPIRV: AtomicCompareExchange [[Int]] [[Res:[0-9]+]] [[Pointer]] [[MemScope_Device]]
; CHECK-SPIRV-SAME:                  [[MemSemEqual_SeqCst]] [[MemSemUnequal_Acquire]] [[Value]] [[Comparator]]
; CHECK-SPIRV: IEqual {{[0-9]+}} [[Success:[0-9]+]] [[Res]] [[Comparator]]
; CHECK-SPIRV: BranchConditional [[Success]]

; CHECK-SPIRV: Store [[Value_ptr]] [[Res]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind
define dso_local spir_func void @test(i32* %ptr, i32* %value_ptr, i32 %comparator) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* %value_ptr, align 4
  %1 = cmpxchg i32* %ptr, i32 %comparator, i32 %0 seq_cst acquire
  %2 = extractvalue { i32, i1 } %1, 1
  br i1 %2, label %cmpxchg.continue, label %cmpxchg.store_expected

cmpxchg.store_expected:                           ; preds = %entry
  %3 = extractvalue { i32, i1 } %1, 0
  store i32 %3, i32* %value_ptr, align 4
  br label %cmpxchg.continue

cmpxchg.continue:                                 ; preds = %cmpxchg.store_expected, %entry
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git cfebd7774229885e7ec88ae9ef1f4ae819cce1d2)"}
