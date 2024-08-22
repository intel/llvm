; -- Per-kernel split
; RUN: sycl-post-link -split=kernel -emit-only-kernels-as-entry-points -S < %s -o %tC.table
; RUN: FileCheck %s -input-file=%tC_0.ll --check-prefixes CHECK-A0
; RUN: FileCheck %s -input-file=%tC_1.ll --check-prefixes CHECK-A1

define dso_local amdgpu_kernel void @Kernel1() {
  ret void
}

define dso_local amdgpu_kernel void @Kernel2() {
  ret void
}

; CHECK-A0: define dso_local amdgpu_kernel void @Kernel2()
; CHECK-A0-NOT: define dso_local amdgpu_kernel void @Kernel1()
; CHECK-A1-NOT: define dso_local amdgpu_kernel void @Kernel2()
; CHECK-A1: define dso_local amdgpu_kernel void @Kernel1()
