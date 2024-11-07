; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_min_max %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-DAG: Extension "SPV_EXT_shader_atomic_float_min_max"
; CHECK-DAG: Capability AtomicFloat64MinMaxEXT
; CHECK: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-DAG: Constant [[Int]] [[Scope_Device:[0-9]+]] 1 {{$}}
; CHECK-DAG: Constant [[Int]] [[MemSem_SequentiallyConsistent:[0-9]+]] 16
; CHECK: TypeFloat [[Double:[0-9]+]] 64
; CHECK: Variable {{[0-9]+}} [[DoublePointer:[0-9]+]]
; CHECK: Constant [[Double]] [[DoubleValue:[0-9]+]] 0 1078263808

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@f = common dso_local local_unnamed_addr addrspace(1) global double 0.000000e+00, align 8

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw_fadd() local_unnamed_addr #0 {
entry:
 %0 = atomicrmw fmin ptr addrspace(1) @f, double 42.000000e+00 seq_cst
; CHECK: AtomicFMinEXT [[Double]] {{[0-9]+}} [[DoublePointer]] [[Scope_Device]] [[MemSem_SequentiallyConsistent]] [[DoubleValue]]
 %1 = atomicrmw fmax ptr addrspace(1) @f, double 42.000000e+00 seq_cst
; CHECK: AtomicFMaxEXT [[Double]] {{[0-9]+}} [[DoublePointer]] [[Scope_Device]] [[MemSem_SequentiallyConsistent]] [[DoubleValue]]

  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
