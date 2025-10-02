; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_shader_atomic_bfloat16,+SPV_KHR_bfloat16 %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-DAG: Extension "SPV_INTEL_shader_atomic_bfloat16"
; CHECK-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-DAG: Capability AtomicBFloat16AddINTEL
; CHECK-DAG: Capability BFloat16TypeKHR
; CHECK: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-DAG: Constant [[Int]] [[Scope_CrossDevice:[0-9]+]] 0 {{$}}
; CHECK-DAG: Constant [[Int]] [[MemSem_SequentiallyConsistent:[0-9]+]] 16
; CHECK: TypeFloat [[BFloat:[0-9]+]] 16 0
; CHECK: Variable {{[0-9]+}} [[BFloatPointer:[0-9]+]]
; CHECK: Constant [[BFloat]] [[BFloatValue:[0-9]+]] 16936

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@f = common dso_local local_unnamed_addr addrspace(1) global bfloat 0.000000e+00, align 8

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw_fadd() local_unnamed_addr #0 {
entry:
 %0 = atomicrmw fadd ptr addrspace(1) @f, bfloat 42.000000e+00 seq_cst
; CHECK: AtomicFAddEXT [[BFloat]] {{[0-9]+}} [[BFloatPointer]] [[Scope_CrossDevice]] [[MemSem_SequentiallyConsistent]] [[BFloatValue]]

  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
