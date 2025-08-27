;;===----------------------------------------------------------------------===;;
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
;;===----------------------------------------------------------------------===;;

define float @_Z21__spirv_AtomicFAddEXTPfiif(ptr noundef %p, i32 noundef %scope, i32 noundef %semantics, float noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr %p, float %val seq_cst, align 4
  ret float %0
}

define float @_Z21__spirv_AtomicFAddEXTPU3AS1fiif(ptr addrspace(1) noundef %p, i32 noundef %scope, i32 noundef %semantics, float noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr addrspace(1) %p, float %val seq_cst, align 4
  ret float %0
}

define float @_Z21__spirv_AtomicFAddEXTPU3AS3fiif(ptr addrspace(3) noundef %p, i32 noundef %scope, i32 noundef %semantics, float noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr addrspace(3) %p, float %val seq_cst, align 4
  ret float %0
}

define double @_Z21__spirv_AtomicFAddEXTPdiid(ptr noundef %p, i32 noundef %scope, i32 noundef %semantics, double noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr %p, double %val seq_cst, align 8
  ret double %0
}

define double @_Z21__spirv_AtomicFAddEXTPU3AS1diid(ptr addrspace(1) noundef %p, i32 noundef %scope, i32 noundef %semantics, double noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr addrspace(1) %p, double %val seq_cst, align 8
  ret double %0
}

define double @_Z21__spirv_AtomicFAddEXTPU3AS3diid(ptr addrspace(3) noundef %p, i32 noundef %scope, i32 noundef %semantics, double noundef %val) nounwind alwaysinline {
entry:
  %0 = atomicrmw fadd ptr addrspace(3) %p, double %val seq_cst, align 8
  ret double %0
}
