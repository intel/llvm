; RUN: opt -S -sycllowerwglocalmemory -bugpoint-enable-legacy-pm < %s | FileCheck %s
; RUN: opt -S -passes=sycllowerwglocalmemory < %s | FileCheck %s

; CHECK-NOT: __sycl_dynamicLocalMemoryPlaceholder

target triple = "nvptx64-nvidia-cuda"

; CHECK: @__sycl_dynamicLocalMemoryPlaceholder_GV = external addrspace(3) global [0 x i8], align 128

; Function Attrs: convergent norecurse
; CHECK: @_ZTS7KernelA(ptr addrspace(1) %0)
define void @_ZTS7KernelA(ptr addrspace(1) %0) local_unnamed_addr #0 !kernel_arg_addr_space !5 {
entry:
; CHECK: getelementptr inbounds i8, ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder_GV
  %1 = tail call spir_func ptr addrspace(3) @__sycl_dynamicLocalMemoryPlaceholder(i64 128) #1
  %2 = getelementptr inbounds i8, ptr addrspace(3) %1, i64 4
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
