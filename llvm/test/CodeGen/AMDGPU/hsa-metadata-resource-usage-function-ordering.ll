; Note: uses a randomly selected assumed external call stack size so that the
; test assertions are unlikely to succeed by accident.

; RUN: llc -amdgpu-assume-external-call-stack-size=5310 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -enable-misched=0 -filetype=asm -o - < %s | FileCheck --check-prefixes CHECK,GFX7 %s
; RUN: llc -amdgpu-assume-external-call-stack-size=5310 -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -enable-misched=0 -filetype=asm -o - < %s | FileCheck --check-prefixes CHECK,GFX8 %s
; RUN: llc -amdgpu-assume-external-call-stack-size=5310 -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -filetype=asm -o - < %s | FileCheck --check-prefixes CHECK,GFX9 %s
; RUN: llc -amdgpu-assume-external-call-stack-size=5310 -mattr=-xnack -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -enable-misched=0 -filetype=asm -o - < %s | FileCheck --check-prefixes CHECK,GFX10 %s

; CHECK-LABEL: amdhsa.kernels

; test a kernel without an external call that occurs before its callee in the module
; CHECK-LABEL: test1
; CHECK:     .private_segment_fixed_size: 20

; GFX7: .sgpr_count:     37
; GFX7: .sgpr_spill_count: 0
; GFX7: .vgpr_count:     4
; GFX7: .vgpr_spill_count: 0

; GFX8:     .sgpr_count:     39
; GFX8:     .sgpr_spill_count: 0
; GFX8:     .vgpr_count:     4
; GFX8:     .vgpr_spill_count: 0

; GFX9:     .sgpr_count:     39
; GFX9:     .sgpr_spill_count: 0
; GFX9:     .vgpr_count:     4
; GFX9:     .vgpr_spill_count: 0

; GFX10:     .sgpr_count:     33
; GFX10:     .sgpr_spill_count: 0
; GFX10:     .vgpr_count:     4
; GFX10:     .vgpr_spill_count: 0
define amdgpu_kernel void @test1(ptr %x) #1 {
  %1 = load volatile float, ptr %x
  %2 = call float @f(float %1)
  store volatile float %2, ptr %x
  ret void
}

define internal float @f(float %arg0) #1 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 3.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %add = fadd float %arg0, %val
  ret float %add
}

; test a kernel without an external call that occurs after its callee in the module
; CHECK-LABEL: test2
; CHECK:     .private_segment_fixed_size: 20

; GFX7:     .sgpr_count:     37
; GFX7:     .sgpr_spill_count: 0
; GFX7:     .vgpr_count:     4
; GFX7:     .vgpr_spill_count: 0

; GFX8:     .sgpr_count:     39
; GFX8:     .sgpr_spill_count: 0
; GFX8:     .vgpr_count:     4
; GFX8:     .vgpr_spill_count: 0

; GFX9:     .sgpr_count:     39
; GFX9:     .sgpr_spill_count: 0
; GFX9:     .vgpr_count:     4
; GFX9:     .vgpr_spill_count: 0

; GFX10:     .sgpr_count:     33
; GFX10:     .sgpr_spill_count: 0
; GFX10:     .vgpr_count:     4
; GFX10:     .vgpr_spill_count: 0
define amdgpu_kernel void @test2(ptr %x) {
  %1 = load volatile float, ptr %x
  %2 = call float @f(float %1)
  store volatile float %2, ptr %x
  ret void
}

; test a kernel with an external call that occurs before its callee in the module
; CHECK-LABEL: test3
; CHECK:     .private_segment_fixed_size: 5310

; GFX7:     .sgpr_count:     37
; GFX7:     .sgpr_spill_count: 0
; GFX7:     .vgpr_count:     32
; GFX7:     .vgpr_spill_count: 0

; GFX8:     .sgpr_count:     39
; GFX8:     .sgpr_spill_count: 0
; GFX8:     .vgpr_count:     32
; GFX8:     .vgpr_spill_count: 0

; GFX9:     .sgpr_count:     39
; GFX9:     .sgpr_spill_count: 0
; GFX9:     .vgpr_count:     32
; GFX9:     .vgpr_spill_count: 0

; GFX10:     .sgpr_count:     35
; GFX10:     .sgpr_spill_count: 0
; GFX10:     .vgpr_count:     32
; GFX10:     .vgpr_spill_count: 0
define amdgpu_kernel void @test3() {
  call void @g()
  ret void
}

declare void @g() #0

; test a kernel without an external call that occurs after its callee in the module
; CHECK-LABEL: test4
; CHECK:     .private_segment_fixed_size: 5310

; GFX7:     .sgpr_count:     37
; GFX7:     .sgpr_spill_count: 0
; GFX7:     .vgpr_count:     32
; GFX7:     .vgpr_spill_count: 0

; GFX8:     .sgpr_count:     39
; GFX8:     .sgpr_spill_count: 0
; GFX8:     .vgpr_count:     32
; GFX8:     .vgpr_spill_count: 0

; GFX9:     .sgpr_count:     39
; GFX9:     .sgpr_spill_count: 0
; GFX9:     .vgpr_count:     32
; GFX9:     .vgpr_spill_count: 0

; GFX10:     .sgpr_count:     35
; GFX10:     .sgpr_spill_count: 0
; GFX10:     .vgpr_count:     32
; GFX10:     .vgpr_spill_count: 0
define amdgpu_kernel void @test4() {
  call void @g()
  ret void
}

attributes #0 = { norecurse }
attributes #1 = { norecurse "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
