; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_cache_controls %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#Func:]] "test"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#FuncGEP:]] "test_gep"
; CHECK-SPIRV-DAG: TypeInt [[#Int32:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32]] [[#Zero:]] 0
; CHECK-SPIRV-DAG: Decorate [[#GEP1:]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV-DAG: Decorate [[#GEP1]] CacheControlLoadINTEL 0 3
; CHECK-SPIRV-DAG: Decorate [[#GEP2:]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV-DAG: Decorate [[#GEP2]] CacheControlLoadINTEL 0 3

; CHECK-SPIRV: Function [[#]] [[#Func]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Buffer:]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEP1]] [[#Buffer]] [[#Zero]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#]] [[#GEP1]]

; CHECK-SPIRV: Function [[#]] [[#FuncGEP]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Buffer:]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEP3:]] [[#Buffer]] [[#Zero]]
; CHECK-SPIRV: Bitcast [[#]] [[#BitCast:]] [[#GEP3]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEP2]] [[#BitCast:]] [[#Zero]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#]] [[#GEP2]]

; CHECK-LLVM: define spir_kernel void @test(ptr addrspace(1) %[[Param1:[a-z0-9_.]+]])
; CHECK-LLVM: %[[#GEP:]] = getelementptr i8, ptr addrspace(1) %[[Param1]], i32 0, !spirv.Decorations ![[#MD:]]
; CHECK-LLVM: call spir_func void @foo(ptr addrspace(1) %[[#GEP:]])

; CHECK-LLVM: define spir_kernel void @test_gep(ptr addrspace(1) %[[Param2:[a-z0-9_.]+]])
; CHECK-LLVM: %[[#GEP1:]] = getelementptr ptr addrspace(1), ptr addrspace(1) %[[Param2]], i32 0
; CHECK-LLVM: %[[#BitCast:]] = bitcast ptr addrspace(1) %[[#GEP1]] to ptr addrspace(1)
; CHECK-LLVM: %[[#GEP2:]] = getelementptr i8, ptr addrspace(1) %[[#BitCast]], i32 0, !spirv.Decorations ![[#MD]]
; CHECK-LLVM: call spir_func void @foo(ptr addrspace(1) %[[#GEP2:]])

; CHECK-LLVM: ![[#MD]] = !{![[#Dec1:]], ![[#Dec2:]]}
; CHECK-LLVM: ![[#Dec1]] = !{i32 6442, i32 1, i32 1}
; CHECK-LLVM: ![[#Dec2]] = !{i32 6442, i32 0, i32 3}

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %buffer1) {
entry:
  call void @foo(ptr addrspace(1) %buffer1), !spirv.DecorationCacheControlINTEL !3
  ret void
}

define spir_kernel void @test_gep(ptr addrspace(1) %buffer1) {
entry:
  %0 = getelementptr ptr addrspace(1), ptr addrspace(1) %buffer1, i32 0
  call void @foo(ptr addrspace(1) %0), !spirv.DecorationCacheControlINTEL !3
  ret void
}

declare void @foo(ptr addrspace(1))

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4, !5}
!4 = !{i32 6442, i32 0, i32 3, i32 0}
!5 = !{i32 6442, i32 1, i32 1, i32 0}
