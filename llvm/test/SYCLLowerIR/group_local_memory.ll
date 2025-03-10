; RUN: opt -S -sycllowerwglocalmemory -bugpoint-enable-legacy-pm < %s | FileCheck %s
; RUN: opt -S -passes=sycllowerwglocalmemory < %s | FileCheck %s

; CHECK-DAG: [[WGLOCALMEM_1:@WGLocalMem.*]] = internal addrspace(3) global [128 x i8] poison, align 4
; CHECK-DAG: [[WGLOCALMEM_2:@WGLocalMem.*]] = internal addrspace(3) global [4 x i8] poison, align 4
; CHECK-DAG: [[WGLOCALMEM_3:@WGLocalMem.*]] = internal addrspace(3) global [256 x i8] poison, align 8

; CHECK-NOT: __sycl_allocateLocalMemory

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS7KernelA() local_unnamed_addr #0 {
entry:
  %0 = tail call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 128, i64 4) #2
  %1 = getelementptr inbounds i8, ptr addrspace(3) %0, i64 4
  ; CHECK: ptr addrspace(3) [[WGLOCALMEM_1]]
  %2 = tail call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 4, i64 4) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64, i64) local_unnamed_addr #1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS7KernelB() local_unnamed_addr #0 {
entry:
  %0 = tail call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 256, i64 8) #2
  ret void
}

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0"}
