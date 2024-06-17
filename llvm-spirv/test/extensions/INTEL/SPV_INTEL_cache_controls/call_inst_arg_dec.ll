; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate [[#GEP1:]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV: Decorate [[#GEP2:]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV: Constant [[#Int32Ty]] [[#Zero:]] 0
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEP1]] [[#]] [[#Zero]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEP2]] [[#]] [[#Zero]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#]] [[#GEP1]] [[#GEP2]]

; CHECK-LLVM: %[[#GEP1:]] = getelementptr i8, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache1:]]
; CHECK-LLVM: %[[#GEP2:]] = getelementptr i8, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache2:]]
; CHECK-LLVM: call spir_func void @foo(ptr addrspace(1) %[[#GEP1]], ptr addrspace(1) %[[#GEP2]])
; CHECK-LLVM: ![[#Cache1]] = !{![[#LoadCache:]]}
; CHECK-LLVM: ![[#LoadCache]] = !{i32 6442, i32 0, i32 1}
; CHECK-LLVM: ![[#Cache2]] = !{![[#StoreCache:]]}
; CHECK-LLVM: ![[#StoreCache]] = !{i32 6443, i32 0, i32 1}

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %buffer1, ptr addrspace(1) %buffer2) {
entry:
  call void @foo(ptr addrspace(1) %buffer1, ptr addrspace(1) %buffer2), !spirv.DecorationCacheControlINTEL !3
  ret void
}

declare void @foo(ptr addrspace(1), ptr addrspace(1))

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4, !5}
!4 = !{i32 6442, i32 0, i32 1, i32 0}
!5 = !{i32 6443, i32 0, i32 1, i32 1}

