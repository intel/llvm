; RUN: opt -S -passes=sycllowerwglocalmemory < %s | FileCheck %s

; Test that ptx_kernel (CUDA) kernels using dynamic work-group local memory are
; lowered correctly. Specifically, calls to __sycl_dynamicLocalMemoryPlaceholder
; must be replaced by references to a real shared-memory global, and each
; kernel must be stamped with the "sycl-work-group-scratch" attribute so that
; the host-side launch code knows to wire up the dynamic shared memory size
; when invoking cudaLaunchKernel.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: @__sycl_dynamicLocalMemoryPlaceholder_GV = external local_unnamed_addr addrspace(3) global [0 x i8]

; CHECK-LABEL: @kernel_a(
define ptx_kernel void @kernel_a(ptr addrspace(1) %0) #0 {
entry:
; CHECK-NOT: call {{.*}} @__sycl_dynamicLocalMemoryPlaceholder
; CHECK: getelementptr inbounds i8, ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder_GV
  %1 = call ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64 128) #1
  %2 = getelementptr inbounds i8, ptr addrspace(3) %1, i64 4
  store i8 0, ptr addrspace(3) %2
  ret void
}

; CHECK-LABEL: @kernel_b(
define ptx_kernel void @kernel_b(ptr addrspace(1) %0) #1 {
entry:
  %1 = call ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64 64) #1
  ret void
}

declare ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64) #1

attributes #0 = { convergent norecurse }
attributes #1 = { convergent }

; Both ptx_kernel functions must receive "sycl-work-group-scratch".
; CHECK-DAG: attributes #{{[0-9]+}} = { convergent norecurse "sycl-work-group-scratch" }
; CHECK-DAG: attributes #{{[0-9]+}} = { convergent "sycl-work-group-scratch" }