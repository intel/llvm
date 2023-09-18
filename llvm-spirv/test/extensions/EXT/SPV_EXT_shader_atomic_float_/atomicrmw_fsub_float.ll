; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_shader_atomic_float_add %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck --check-prefix=CHECK-SPIRV %s

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM

; CHECK-SPIRV-DAG: Extension "SPV_EXT_shader_atomic_float_add"
; CHECK-SPIRV-DAG: Capability AtomicFloat32AddEXT
; CHECK-SPIRV: TypeInt [[Int:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: Constant [[Int]] [[Scope_Device:[0-9]+]] 1 {{$}}
; CHECK-SPIRV-DAG: Constant [[Int]] [[MemSem_SequentiallyConsistent:[0-9]+]] 16
; CHECK-SPIRV: TypeFloat [[Float:[0-9]+]] 32
; CHECK-SPIRV: Variable {{[0-9]+}} [[FPPointer:[0-9]+]]
; CHECK-SPIRV: Constant [[Float]] [[FPValue:[0-9]+]] 1109917696

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@f = common dso_local local_unnamed_addr addrspace(1) global float 0.000000e+00, align 4

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw_fadd() local_unnamed_addr #0 {
entry:
 %0 = atomicrmw fsub ptr addrspace(1) @f, float 42.000000e+00 seq_cst
; CHECK-SPIRV: FNegate [[Float]] [[NegateValue:[0-9]+]] [[FPValue]]
; CHECK-SPIRV: AtomicFAddEXT [[Float]] {{[0-9]+}} [[FPPointer]] [[Scope_Device]] [[MemSem_SequentiallyConsistent]] [[NegateValue]]
; CHECK-LLVM: [[FNegateLLVM:%[0-9]+]] = fneg float 4.200000e+01
; CHECK-LLVM: call spir_func float {{.*}}atomic_add{{.*}}(ptr addrspace(1) @f, float [[FNegateLLVM]])
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
