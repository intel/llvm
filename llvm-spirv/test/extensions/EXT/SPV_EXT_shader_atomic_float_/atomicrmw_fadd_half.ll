; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float16_add %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-DAG: Extension "SPV_EXT_shader_atomic_float16_add"
; CHECK-DAG: Extension "SPV_EXT_shader_atomic_float_add"
; CHECK-DAG: Capability AtomicFloat16AddEXT
; CHECK: TypeInt [[TypeIntID:[0-9]+]] 32 0
; CHECK-DAG: Constant [[TypeIntID]] [[ScopeDevice:[0-9]+]] 1 {{$}}
; CHECK-DAG: Constant [[TypeIntID]] [[MemSem_SequentiallyConsistent:[0-9]+]] 16
; CHECK: TypeFloat [[TypeFloatHalfID:[0-9]+]] 16
; CHECK: Variable {{[0-9]+}} [[HalfPointer:[0-9]+]]
; CHECK: Constant [[TypeFloatHalfID]] [[HalfValue:[0-9]+]] 20800

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@f = common dso_local local_unnamed_addr addrspace(1) global half 0.000000e+00, align 4

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw_fadd() local_unnamed_addr #0 {
entry:
 %0 = atomicrmw fadd ptr addrspace(1) @f, half 42.000000e+00 seq_cst
; CHECK: AtomicFAddEXT [[TypeFloatHalfID]] {{[0-9]+}} [[HalfPointer]] [[ScopeDevice]] [[MemSem_SequentiallyConsistent]] [[HalfValue]]

  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
