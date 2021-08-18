; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_shader_atomic_float_add -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-CL,CHECK-LLVM-CL12

; RUN: llvm-spirv --spirv-target-env=CL2.0 -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-CL,CHECK-LLVM-CL20

; RUN: llvm-spirv --spirv-target-env=SPV-IR -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-SPV

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability AtomicFloat32AddEXT
; CHECK-SPIRV: Capability AtomicFloat64AddEXT
; CHECK-SPIRV: Extension "SPV_EXT_shader_atomic_float_add"
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_32:[0-9]+]] 32
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_64:[0-9]+]] 64

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func float @_Z14AtomicFloatIncRf(float addrspace(4)* align 4 dereferenceable(4) %Arg) local_unnamed_addr #0 {
entry:
  %0 = addrspacecast float addrspace(4)* %Arg to float addrspace(1)*
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_32]]
  ; CHECK-LLVM-CL12: call spir_func float @[[FLOAT_FUNC_NAME:_Z10atomic_add[[:alnum:]]+ff]]({{.*}})
  ; CHECK-LLVM-CL20: call spir_func float @[[FLOAT_FUNC_NAME:_Z25atomic_fetch_add_explicit[[:alnum:]]+_Atomicff[a-zA-Z0-9_]+]]({{.*}})
  ; CHECK-LLVM-SPV: call spir_func float @[[FLOAT_FUNC_NAME:_Z21__spirv_AtomicFAddEXT[[:alnum:]]+fiif]]({{.*}})
  %call3.i.i = tail call spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(float addrspace(1)* %0, i32 1, i32 896, float 1.000000e+00) #2
  ret float %call3.i.i
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(float addrspace(1)*, i32, i32, float) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare {{.*}}spir_func float @[[FLOAT_FUNC_NAME]](float

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func double @_Z15AtomicDoubleIncRd(double addrspace(4)* align 8 dereferenceable(8) %Arg) local_unnamed_addr #0 {
entry:
  %0 = addrspacecast double addrspace(4)* %Arg to double addrspace(1)*
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_64]]
  ; CHECK-LLVM-CL12: call spir_func double @[[DOUBLE_FUNC_NAME:_Z10atomic_add[[:alnum:]]+dd]]({{.*}})
  ; CHECK-LLVM-CL20: call spir_func double @[[DOUBLE_FUNC_NAME:_Z25atomic_fetch_add_explicit[[:alnum:]]+_Atomicdd[a-zA-Z0-9_]+]]({{.*}})
  ; CHECK-LLVM-SPV: call spir_func double @[[DOUBLE_FUNC_NAME:_Z21__spirv_AtomicFAddEXT[[:alnum:]]+diid]]({{.*}})
  %call3.i.i = tail call spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(double addrspace(1)* %0, i32 1, i32 896, double 1.000000e+00) #2
  ret double %call3.i.i
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(double addrspace(1)*, i32, i32, double) local_unnamed_addr #1
; CHECK-LLVM-SPV: declare {{.*}}spir_func double @[[DOUBLE_FUNC_NAME]](double

; CHECK-LLVM-CL: declare {{.*}}spir_func float @[[FLOAT_FUNC_NAME]](float
; CHECK-LLVM-CL: declare {{.*}}spir_func double @[[DOUBLE_FUNC_NAME]](double

attributes #0 = { convergent norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0"}
