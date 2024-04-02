; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_shader_atomic_float_min_max -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv --spirv-target-env=CL2.0 -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-CL20

; RUN: llvm-spirv --spirv-target-env=SPV-IR -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-SPV

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Capability AtomicFloat32MinMaxEXT
; CHECK-SPIRV: Capability AtomicFloat64MinMaxEXT
; CHECK-SPIRV: Extension "SPV_EXT_shader_atomic_float_min_max"
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_32:[0-9]+]] 32
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_64:[0-9]+]] 64

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func void @test_float(ptr addrspace(1) %a) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: 7 AtomicFMaxEXT [[TYPE_FLOAT_32]]
  ; CHECK-LLVM-CL20: call spir_func float @[[FLOAT_FUNC_NAME:_Z25atomic_fetch_max_explicit[[:alnum:]]+_Atomicff[a-zA-Z0-9_]+]]({{.*}})
  ; CHECK-LLVM-SPV: call spir_func float @[[FLOAT_FUNC_NAME:_Z21__spirv_AtomicFMaxEXT[[:alnum:]]+fiif]]({{.*}})
  %call = tail call spir_func float @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicff12memory_order(ptr addrspace(1) %a, float 0.000000e+00, i32 0) #2
  ret void
}

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicff12memory_order(ptr addrspace(1), float, i32) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare {{.*}}spir_func float @[[FLOAT_FUNC_NAME]]({{.*}}float

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func void @test_double(ptr addrspace(1) %a) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: 7 AtomicFMaxEXT [[TYPE_FLOAT_64]]
  ; CHECK-LLVM-CL20: call spir_func double @[[DOUBLE_FUNC_NAME:_Z25atomic_fetch_max_explicit[[:alnum:]]+_Atomicdd[a-zA-Z0-9_]+]]({{.*}})
  ; CHECK-LLVM-SPV: call spir_func double @[[DOUBLE_FUNC_NAME:_Z21__spirv_AtomicFMaxEXT[[:alnum:]]+diid]]({{.*}})
  %call = tail call spir_func double @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicdd12memory_order(ptr addrspace(1) %a, double 0.000000e+00, i32 0) #2
  ret void
}

; Function Attrs: convergent
declare spir_func double @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicdd12memory_order(ptr addrspace(1), double, i32) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare {{.*}}spir_func double @[[DOUBLE_FUNC_NAME]]({{.*}}double

; CHECK-LLVM-CL20: declare {{.*}}spir_func float @[[FLOAT_FUNC_NAME]]({{.*}}float
; CHECK-LLVM-CL20: declare {{.*}}spir_func double @[[DOUBLE_FUNC_NAME]]({{.*}}double

attributes #0 = { convergent norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 94aa388f0ce0723bb15503cf41c2c15b288375b9)"}
