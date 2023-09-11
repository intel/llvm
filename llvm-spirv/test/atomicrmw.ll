; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-DAG: Constant [[Int]] [[Scope_Device:[0-9]+]] 1 {{$}}
; CHECK-DAG: Constant [[Int]] [[MemSem_Relaxed:[0-9]+]] 0
; CHECK-DAG: Constant [[Int]] [[MemSem_Acquire:[0-9]+]] 2
; CHECK-DAG: Constant [[Int]] [[MemSem_Release:[0-9]+]] 4 {{$}}
; CHECK-DAG: Constant [[Int]] [[MemSem_AcquireRelease:[0-9]+]] 8
; CHECK-DAG: Constant [[Int]] [[MemSem_SequentiallyConsistent:[0-9]+]] 16
; CHECK-DAG: Constant [[Int]] [[Value:[0-9]+]] 42
; CHECK: TypeFloat [[Float:[0-9]+]] 32
; CHECK: Variable {{[0-9]+}} [[Pointer:[0-9]+]]
; CHECK: Variable {{[0-9]+}} [[FPPointer:[0-9]+]]
; CHECK: Constant [[Float]] [[FPValue:[0-9]+]] 1109917696

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@ui = common dso_local addrspace(1) global i32 0, align 4
@f = common dso_local local_unnamed_addr addrspace(1) global float 0.000000e+00, align 4

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw() local_unnamed_addr #0 {
entry:
  %0 = atomicrmw xchg ptr addrspace(1) @ui, i32 42 acq_rel
; CHECK: AtomicExchange [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_AcquireRelease]] [[Value]]

  %1 = atomicrmw xchg ptr addrspace(1) @f, float 42.000000e+00 seq_cst
; CHECK: AtomicExchange [[Float]] {{[0-9]+}} [[FPPointer]] [[Scope_Device]] [[MemSem_SequentiallyConsistent]] [[FPValue]]

  %2 = atomicrmw add ptr addrspace(1) @ui, i32 42 monotonic
; CHECK: AtomicIAdd [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Relaxed]] [[Value]]

  %3 = atomicrmw sub ptr addrspace(1) @ui, i32 42 acquire
; CHECK: AtomicISub [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Acquire]] [[Value]]

  %4 = atomicrmw or ptr addrspace(1) @ui, i32 42 release
; CHECK: AtomicOr [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Release]] [[Value]]

  %5 = atomicrmw xor ptr addrspace(1) @ui, i32 42 acq_rel
; CHECK: AtomicXor [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_AcquireRelease]] [[Value]]

  %6 = atomicrmw and ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK: AtomicAnd [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_SequentiallyConsistent]] [[Value]]

  %7 = atomicrmw max ptr addrspace(1) @ui, i32 42 monotonic
; CHECK: AtomicSMax [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Relaxed]] [[Value]]

  %8 = atomicrmw min ptr addrspace(1) @ui, i32 42 acquire
; CHECK: AtomicSMin [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Acquire]] [[Value]]

  %9 = atomicrmw umax ptr addrspace(1) @ui, i32 42 release
; CHECK: AtomicUMax [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_Release]] [[Value]]

  %10 = atomicrmw umin ptr addrspace(1) @ui, i32 42 acq_rel
; CHECK: AtomicUMin [[Int]] {{[0-9]+}} [[Pointer]] [[Scope_Device]] [[MemSem_AcquireRelease]] [[Value]]

  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 20c5968e0953d978be4d9d1062801e8758c393b5)"}
