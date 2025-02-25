; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_kernel_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability KernelAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_kernel_attributes"
; CHECK-SPIRV: EntryPoint {{.*}} [[DIM1:[0-9]+]] "Dim1"
; CHECK-SPIRV: EntryPoint {{.*}} [[DIM2:[0-9]+]] "Dim2"
; CHECK-SPIRV: EntryPoint {{.*}} [[DIM3:[0-9]+]] "Dim3"
; CHECK-SPIRV: ExecutionMode [[DIM1]] 5893 4 1 1
; CHECK-SPIRV: ExecutionMode [[DIM2]] 5893 8 4 1
; CHECK-SPIRV: ExecutionMode [[DIM3]] 5893 16 8 4
; CHECK-SPIRV: Function {{.*}} [[DIM1]] {{.*}}
; CHECK-SPIRV: Function {{.*}} [[DIM2]] {{.*}}
; CHECK-SPIRV: Function {{.*}} [[DIM3]] {{.*}}

; CHECK-LLVM: define spir_kernel void @Dim1()
; CHECK-LLVM-SAME: !max_work_group_size ![[MAXWG1:[0-9]+]]

; CHECK-LLVM: define spir_kernel void @Dim2()
; CHECK-LLVM-SAME: !max_work_group_size ![[MAXWG2:[0-9]+]]

; CHECK-LLVM: define spir_kernel void @Dim3()
; CHECK-LLVM-SAME: !max_work_group_size ![[MAXWG3:[0-9]+]]

; CHECK-LLVM: ![[MAXWG1]] = !{i32 4, i32 1, i32 1}
; CHECK-LLVM: ![[MAXWG2]] = !{i32 8, i32 4, i32 1}
; CHECK-LLVM: ![[MAXWG3]] = !{i32 16, i32 8, i32 4}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define spir_kernel void @Dim1() !max_work_group_size !0 {
  ret void
}

define spir_kernel void @Dim2() !max_work_group_size !1 {
  ret void
}

define spir_kernel void @Dim3() !max_work_group_size !2 {
  ret void
}

!0 = !{i32 4}
!1 = !{i32 8, i32 4}
!2 = !{i32 16, i32 8, i32 4}
