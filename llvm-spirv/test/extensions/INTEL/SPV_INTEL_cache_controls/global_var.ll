; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Store [[StorePtr:[0-9]+]]

; CHECK-SPIRV-DAG: Decorate [[StorePtr]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV-DAG: Decorate [[StorePtr]] CacheControlStoreINTEL 1 3

; CHECK-LLVM: @p = common addrspace(1) global i32 0, align 4, !spirv.Decorations [[GlobalMD:![0-9]+]]
; CHECK-LLVM: store i32 0, ptr addrspace(1) @p, align 4

; CHECK-LLVM-DAG: [[CC0:![0-9]+]] = !{i32 6443, i32 0, i32 1}
; CHECK-LLVM-DAG: [[CC1:![0-9]+]] = !{i32 6443, i32 1, i32 3}
; CHECK-LLVM-DAG: [[GlobalMD]] = {{.*}}[[CC0]]{{.*}}[[CC1]]

target triple = "spir64-unknown-unknown"

@p = common addrspace(1) global i32 0, align 4, !spirv.Decorations !3

define spir_kernel void @test() {
entry:
  store i32 0, i32 addrspace(1)* @p, align 4
  ret void
}

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4, !5}
!4 = !{i32 6443, i32 0, i32 1}  ; {CacheControlStoreINTEL, CacheLevel=0, WriteThrough}
!5 = !{i32 6443, i32 1, i32 3}  ; {CacheControlStoreINTEL, CacheLevel=1, Streaming}
