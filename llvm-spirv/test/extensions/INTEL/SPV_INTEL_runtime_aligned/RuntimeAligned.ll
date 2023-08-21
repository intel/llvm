; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_runtime_aligned -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability RuntimeAlignedAttributeINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_runtime_aligned"
; CHECK-SPIRV: Name [[#ARGA:]] "a"
; CHECK-SPIRV: Name [[#ARGB:]] "b"
; CHECK-SPIRV: Name [[#ARGC:]] "c"
; CHECK-SPIRV: Name [[#ARGD:]] "d"
; CHECK-SPIRV: Name [[#ARGE:]] "e"
; CHECK-SPIRV: Decorate [[#ARGA]] RuntimeAlignedINTEL 1
; CHECK-SPIRV-NOT: Decorate [[#ARGB]] RuntimeAlignedINTEL [[#]]
; CHECK-SPIRV: Decorate [[#ARGC]] RuntimeAlignedINTEL 1
; CHECK-SPIRV-NOT: Decorate [[#ARGD]] RuntimeAlignedINTEL [[#]]
; CHECK-SPIRV-NOT: Decorate [[#ARGE]] RuntimeAlignedINTEL [[#]]

; CHECK-SPIRV: Function
; CHECK-SPIRV: FunctionParameter [[#]] [[#ARGA]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#ARGB]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#ARGC]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#ARGD]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#ARGE]]

; CHECK-LLVM: define spir_kernel void @test{{.*}} !kernel_arg_runtime_aligned ![[RTALIGN_MD:[0-9]+]] {{.*}}
; CHECK-LLVM: ![[RTALIGN_MD]] = !{i1 true, i1 false, i1 true, i1 false, i1 false}

; ModuleID = 'runtime_aligned.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(i32 addrspace(1)* %a, float addrspace(1)* %b, i32 addrspace(1)* %c, i32 %d, i32 %e) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !9 !kernel_arg_runtime_aligned !10 {
entry:
  ret void
}

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 2, i32 2}
!1 = !{i32 0, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 6, i16 14}
!5 = !{i32 1, i32 1, i32 1, i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none", !"none"}
!7 = !{!"int*", !"float*", !"int*"}
!8 = !{!"", !"", !"", !"", !""}
!9 = !{!"int*", !"float*", !"int*", !"int", !"int"}
!10 = !{i1 true, i1 false, i1 true, i1 false, i1 false}
