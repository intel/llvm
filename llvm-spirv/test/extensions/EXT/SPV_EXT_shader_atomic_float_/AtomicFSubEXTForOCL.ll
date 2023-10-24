;; Check that atomic_fetch_sub is translated to OpAtomicFAddEXT with negative
;; value operand
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_shader_atomic_float_add -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv --spirv-target-env=CL2.0 -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-CL20

; RUN: llvm-spirv --spirv-target-env=SPV-IR -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-SPV

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Capability AtomicFloat32AddEXT
; CHECK-SPIRV: Capability AtomicFloat64AddEXT
; CHECK-SPIRV: Extension "SPV_EXT_shader_atomic_float_add"
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_32:[0-9]+]] 32
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_64:[0-9]+]] 64
;; Check float operand of atomic_fetch_sub is handled correctly
; CHECK-SPIRV: Constant [[TYPE_FLOAT_32]] [[NEGATIVE_229:[0-9]+]] 3278176256
; CHECK-SPIRV: Constant [[TYPE_FLOAT_64]] [[NEGATIVE_334:[0-9]+]] 0 3228884992


; Function Attrs: convergent norecurse nounwind
define dso_local spir_func void @test_atomic_float(ptr addrspace(1) %a) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_32]] 13 7 10 11 [[NEGATIVE_229]]
  ; CHECK-LLVM-CL20: call spir_func float @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4) %a.as, float -2.290000e+02, i32 0, i32 1) #0
  ; CHECK-LLVM-SPV: call spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fiif(ptr addrspace(1) %a, i32 2, i32 0, float -2.290000e+02) #0
  %call2 = tail call spir_func float @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(ptr addrspace(1) noundef %a, float noundef 2.290000e+02, i32 noundef 0, i32 noundef 1) #2
  ret void
}

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(ptr addrspace(1) noundef, float noundef, i32 noundef, i32 noundef) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fiif(ptr addrspace(1), i32, i32, float) #0

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func void @test_atomic_double(ptr addrspace(1) %a) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_64]] 21 18 10 11 [[NEGATIVE_334]]
  ; CHECK-LLVM-CL20: call spir_func double @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(4) %a.as, double -3.340000e+02, i32 0, i32 1) #0
  ; CHECK-LLVM-SPV: call spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1diid(ptr addrspace(1) %a, i32 2, i32 0, double -3.340000e+02) #0
  %call = tail call spir_func double @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(1) noundef %a, double noundef 3.340000e+02, i32 noundef 0, i32 noundef 1) #2
  ret void
}
; Function Attrs: convergent
declare spir_func double @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(1) noundef, double noundef, i32 noundef, i32 noundef) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1diid(ptr addrspace(1), i32, i32, double) #0

; CHECK-LLVM-CL20: declare spir_func float @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4), float, i32, i32) #0
; CHECK-LLVM-CL20: declare spir_func double @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(4), double, i32, i32) #0

attributes #0 = { convergent norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
