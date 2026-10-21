; Check that all cache control decorations attached to a single memory
; instruction are translated, not only the first two.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Decorate [[#StoreGEP:]] CacheControlStoreINTEL 0 2
; CHECK-SPIRV-DAG: Decorate [[#StoreGEP]] CacheControlStoreINTEL 1 3
; CHECK-SPIRV-DAG: Decorate [[#StoreGEP]] CacheControlStoreINTEL 2 3
; CHECK-SPIRV-DAG: Decorate [[#StoreGEP]] CacheControlStoreINTEL 3 1
; CHECK-SPIRV-DAG: Decorate [[#LoadGEP:]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV-DAG: Decorate [[#LoadGEP]] CacheControlLoadINTEL 1 0
; CHECK-SPIRV-DAG: Decorate [[#LoadGEP]] CacheControlLoadINTEL 2 0

; CHECK-SPIRV: Store [[#StoreGEP]]
; CHECK-SPIRV: Load [[#]] [[#]] [[#LoadGEP]]

; CHECK-LLVM: getelementptr float, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#StoreMD:]]
; CHECK-LLVM: getelementptr float, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#LoadMD:]]
; CHECK-LLVM-DAG: ![[#StoreMD]] = !{![[#]], ![[#]], ![[#]], ![[#]]}
; CHECK-LLVM-DAG: ![[#LoadMD]] = !{![[#]], ![[#]], ![[#]]}

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %dst, ptr addrspace(1) %src) {
entry:
  store float 1.0, ptr addrspace(1) %dst, align 4, !spirv.DecorationCacheControlINTEL !3
  %v = load float, ptr addrspace(1) %src, align 4, !spirv.DecorationCacheControlINTEL !8
  ret void
}

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 6443, i32 0, i32 2, i32 1}
!5 = !{i32 6443, i32 1, i32 3, i32 1}
!6 = !{i32 6443, i32 2, i32 3, i32 1}
!7 = !{i32 6443, i32 3, i32 1, i32 1}
!8 = !{!9, !10, !11}
!9 = !{i32 6442, i32 0, i32 1, i32 0}
!10 = !{i32 6442, i32 1, i32 0, i32 0}
!11 = !{i32 6442, i32 2, i32 0, i32 0}
