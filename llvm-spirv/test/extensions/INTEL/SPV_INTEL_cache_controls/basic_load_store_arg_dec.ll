; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: TypeInt [[#Int32:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32]] [[#Zero:]] 0
; CHECK-SPIRV-DAG: Decorate [[#Load1GEPPtr:]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV-DAG: Decorate [[#Load2GEPPtr:]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV-DAG: Decorate [[#Store1GEPPtr:]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV-DAG: Decorate [[#Store2GEPPtr:]] CacheControlStoreINTEL 1 1

; CHECK-SPIRV: FunctionParameter [[#]] [[#Buffer:]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#Load1GEPPtr:]] [[#Buffer]] [[#Zero]]
; CHECK-SPIRV: Load [[#]] [[#]] [[#Load1GEPPtr]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#Load2GEPPtr:]] [[#]] [[#Zero]]
; CHECK-SPIRV: Load [[#]] [[#]] [[#Load2GEPPtr]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#Store1GEPPtr:]] [[#]] [[#Zero]]
; CHECK-SPIRV: Store [[#Store1GEPPtr]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#Store2GEPPtr:]] [[#]] [[#Zero]]
; CHECK-SPIRV: Store [[#Store2GEPPtr]]

; CHECK-LLVM: %[[#GEPLoad1:]] = getelementptr i32, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache1:]]
; CHECK-LLVM: load i32, ptr addrspace(1) %[[#GEPLoad1]], align 4
; CHECK-LLVM: %[[#GEPLoad2:]] = getelementptr i32, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache2:]]
; CHECK-LLVM: load i32, ptr addrspace(1) %[[#GEPLoad2]], align 4
; CHECK-LLVM: %[[#GEPStore1:]] = getelementptr i32, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache3:]]
; CHECK-LLVM: store i32 %[[#]], ptr addrspace(1) %[[#GEPStore1]], align 4
; CHECK-LLVM: %[[#GEPStore2:]] = getelementptr i32, ptr addrspace(1) %{{.*}}, i32 0, !spirv.Decorations ![[#Cache4:]]
; CHECK-LLVM: store i32 %[[#]], ptr addrspace(1) %[[#GEPStore2]], align 4
; CHECK-LLVM: ![[#Cache1]] = !{![[#DecLoad1:]]}
; CHECK-LLVM: ![[#DecLoad1]] = !{i32 6442, i32 0, i32 1}
; CHECK-LLVM: ![[#Cache2]] = !{![[#DecLoad2:]]}
; CHECK-LLVM: ![[#DecLoad2]] = !{i32 6442, i32 1, i32 1}
; CHECK-LLVM: ![[#Cache3:]] = !{![[#DecStore1:]]}
; CHECK-LLVM: ![[#DecStore1]] = !{i32 6443, i32 0, i32 1}
; CHECK-LLVM: ![[#Cache4:]] = !{![[#DecStore2:]]}
; CHECK-LLVM: ![[#DecStore2]] = !{i32 6443, i32 1, i32 1}

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %buffer) {
entry:
  %0 = load i32, ptr addrspace(1) %buffer, align 4, !spirv.DecorationCacheControlINTEL !3
  %1 = load i32, ptr addrspace(1) %buffer, align 4, !spirv.DecorationCacheControlINTEL !5
  store i32 %0, ptr addrspace(1) %buffer, align 4, !spirv.DecorationCacheControlINTEL !7
  store i32 %1, ptr addrspace(1) %buffer, align 4, !spirv.DecorationCacheControlINTEL !9
  ret void
}

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4}
!4 = !{i32 6442, i32 0, i32 1, i32 0}
!5 = !{!6}
!6 = !{i32 6442, i32 1, i32 1, i32 0}
!7 = !{!8}
!8 = !{i32 6443, i32 0, i32 1, i32 1}
!9 = !{!10}
!10 = !{i32 6443, i32 1, i32 1, i32 1}
