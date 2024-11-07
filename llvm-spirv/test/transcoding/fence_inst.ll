; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[#UINT:]] 32 0

; 0x0 CrossDevice
; CHECK-SPIRV: Constant [[#UINT]] [[#CD:]] 0

; 0x2 Workgroup
; CHECK-SPIRV: Constant [[#UINT]] [[#ID1:]] 2
; CHECK-SPIRV: Constant [[#UINT]] [[#ID2:]] 4
; CHECK-SPIRV: Constant [[#UINT]] [[#ID3:]] 8
; CHECK-SPIRV: Constant [[#UINT]] [[#ID4:]] 16

; CHECK-SPIRV: MemoryBarrier [[#CD]] [[#ID1]]
; CHECK-SPIRV: MemoryBarrier [[#CD]] [[#ID2]]
; CHECK-SPIRV: MemoryBarrier [[#CD]] [[#ID3]]
; CHECK-SPIRV: MemoryBarrier [[#CD]] [[#ID4]]
; CHECK-SPIRV: MemoryBarrier [[#ID1]] [[#ID2]]


; CHECK-LLVM: define spir_kernel void @fence_test_kernel1{{.*}} #0 {{.*}}
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 0)

; CHECK-LLVM: define spir_kernel void @fence_test_kernel2{{.*}} #0 {{.*}}
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 0)

; CHECK-LLVM: define spir_kernel void @fence_test_kernel3{{.*}} #0 {{.*}}
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 0)

; CHECK-LLVM: define spir_kernel void @fence_test_kernel4{{.*}} #0 {{.*}}
; CHECK-LLVM-NEXT: call spir_func void @_Z9mem_fencej(i32 0)

; ModuleID = 'fence_inst.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: noinline nounwind
define spir_kernel void @fence_test_kernel1(ptr addrspace(1) noalias %s.ascast) {
  fence acquire
  ret void
}

; Function Attrs: noinline nounwind
define spir_kernel void @fence_test_kernel2(ptr addrspace(1) noalias %s.ascast) {
  fence release
  ret void
}

; Function Attrs: noinline nounwind
define spir_kernel void @fence_test_kernel3(ptr addrspace(1) noalias %s.ascast) {
  fence acq_rel
  ret void
}

; Function Attrs: noinline nounwind
define spir_kernel void @fence_test_kernel4(ptr addrspace(1) noalias %s.ascast) {
  fence syncscope("singlethread") seq_cst
  ret void
}

define spir_kernel void @fence_test_kernel5(ptr addrspace(1) noalias %s.ascast) {
  fence syncscope("workgroup") release
  ret void
}

