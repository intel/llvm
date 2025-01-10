; RUN: opt -S -sycllowerwglocalmemory -bugpoint-enable-legacy-pm < %s | FileCheck %s
; RUN: opt -S -passes=sycllowerwglocalmemory < %s | FileCheck %s

; CHECK-NOT: __sycl_dynamicLocalMemoryPlaceholder

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: @__sycl_dynamicLocalMemoryPlaceholder_GV = linkonce_odr local_unnamed_addr addrspace(3) global ptr addrspace(3) undef

; Function Attrs: convergent norecurse
; CHECK: @_ZTS7KernelA(ptr addrspace(1) %0, ptr addrspace(3) noalias "sycl-implicit-local-arg" %[[IMPLICT_ARG:[a-zA-Z0-9]+]]{{.*}} !kernel_arg_addr_space ![[ADDR_SPACE_MD:[0-9]+]]
define weak_odr dso_local spir_kernel void @_ZTS7KernelA(ptr addrspace(1) %0) local_unnamed_addr #0 !kernel_arg_addr_space !5 {
entry:
; CHECK: store ptr addrspace(3) %[[IMPLICT_ARG]], ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder_GV
; CHECK: %[[LD1:[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder_GV
  %1 = tail call spir_func ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64 128) #1
; CHECK: getelementptr inbounds i8, ptr addrspace(3) %[[LD1]]
  %2 = getelementptr inbounds i8, ptr addrspace(3) %1, i64 4
; CHECK: %[[LD2:[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder_GV
  %3 = tail call spir_func ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64 4) #1
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64) local_unnamed_addr #1

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" "sycl-work-group-static"="1" }
attributes #1 = { convergent norecurse }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0"}
!4 = !{}
; ![[ADDR_SPACE_MD]] = !{i32 1, i32 3}
!5 = !{i32 1}
