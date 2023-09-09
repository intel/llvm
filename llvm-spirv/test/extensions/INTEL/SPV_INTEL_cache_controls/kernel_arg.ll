; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate {{[0-9]+}} CacheControlLoadINTEL 0 0
; CHECK-SPIRV: Decorate {{[0-9]+}} CacheControlStoreINTEL 0 1

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: spir_kernel {{.*}} !spirv.ParameterDecorations [[ParamDecID:![0-9]+]]
define spir_kernel void @test(ptr addrspace(1) %dummy, ptr addrspace(1) %buffer) !spirv.ParameterDecorations !3 {
entry:
  %0 = load i32, ptr addrspace(1) %buffer, align 4
  store i32 %0, ptr addrspace(1) %buffer, align 4
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
!4 = !{}
!5 = !{!6, !7}
; CHECK-LLVM: [[ParamDecID]] = !{!{{[0-9]+}}, [[BufferDecID:![0-9]+]]}
; CHECK-LLVM: [[BufferDecID]] = !{[[StoreDecID:![0-9]+]], [[LoadDecID:![0-9]+]]}
; CHECK-LLVM: [[StoreDecID]] = !{i32 6442, i32 0, i32 0}
; CHECK-LLVM: [[LoadDecID]] = !{i32 6443, i32 0, i32 1}
!6 = !{i32 6442, i32 0, i32 0}  ; {CacheControlLoadINTEL,   CacheLevel=0, Uncached}
!7 = !{i32 6443, i32 0, i32 1}  ; {CacheControlStoreINTEL,  CacheLevel=0, WriteThrough}
