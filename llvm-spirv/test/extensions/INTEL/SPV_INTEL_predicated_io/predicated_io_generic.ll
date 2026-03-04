; Generated with:
; source.cl:
; int __spirv_PredicatedLoadINTEL(const __global int* pointer, bool predicate, int default_value);
; int __spirv_PredicatedLoadINTEL(const __global int* pointer, bool predicate, int default_value, int memory_operands);
; void __spirv_PredicatedStoreINTEL(const __global int* pointer, int object, bool predicate);
; void __spirv_PredicatedStoreINTEL(const __global int* pointer, int object, bool predicate, int memory_operands);
;
; void foo(const __global int* load_pointer, __global int* store_pointer, int default_value, int store_object, bool predicate) {
;   const int memory_ops = 0;
;    int result1 = __spirv_PredicatedLoadINTEL(load_pointer, predicate, default_value);
;    int result2 = __spirv_PredicatedLoadINTEL(load_pointer, predicate, default_value, memory_ops);
;    __spirv_PredicatedStoreINTEL(store_pointer, store_object, predicate);
;    __spirv_PredicatedStoreINTEL(store_pointer, store_object, predicate, memory_ops);
; }
; clang -cc1 -cl-std=clc++2021 -triple spir64-unknown-unknown -emit-llvm -finclude-default-header source.cl -o tmp.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_predicated_io
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_predicated_io

; CHECK-SPIRV: Capability PredicatedIOINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_predicated_io"
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const0:]] 0
; CHECK-SPIRV-DAG: TypeVoid [[#VoidTy:]]
; CHECK-SPIRV-DAG: TypePointer [[#IntPtrTy:]] 5 [[#Int32Ty]]
; CHECK-SPIRV-DAG: TypeBool [[#BoolTy:]]
; CHECK-SPIRV: FunctionParameter [[#IntPtrTy]] [[#LoadPtr:]]
; CHECK-SPIRV: FunctionParameter [[#IntPtrTy]] [[#StorePtr:]]
; CHECK-SPIRV: FunctionParameter [[#Int32Ty]] [[#DefaultVal:]]
; CHECK-SPIRV: FunctionParameter [[#Int32Ty]] [[#StoreObj:]]
; CHECK-SPIRV: FunctionParameter [[#BoolTy]] [[#Predicate:]]
; CHECK-SPIRV: PredicatedLoadINTEL [[#Int32Ty]] [[#Result1:]] [[#LoadPtr]] [[#Predicate]] [[#DefaultVal]]
; CHECK-SPIRV: PredicatedLoadINTEL [[#Int32Ty]] [[#Result2:]] [[#LoadPtr]] [[#Predicate]] [[#DefaultVal]] [[#Const0]]
; CHECK-SPIRV: PredicatedStoreINTEL [[#StorePtr]] [[#StoreObj]] [[#Predicate]]
; CHECK-SPIRV: PredicatedStoreINTEL [[#StorePtr]] [[#StoreObj]] [[#Predicate]] [[#Const0]]

; CHECK-LLVM: call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1ibi(ptr addrspace(1) %{{.*}}, i1 %{{.*}}, i32 %{{.*}})
; CHECK-LLVM: call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1ibii(ptr addrspace(1) %{{.*}}, i1 %{{.*}}, i32 %{{.*}}, i32 0)
; CHECK-LLVM: call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1iib(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i1 %{{.*}})
; CHECK-LLVM: call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1iibi(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i32 0)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @foo(ptr addrspace(1) %load_pointer, ptr addrspace(1) %store_pointer, i32  %default_value, i32 %store_object, i1 zeroext %predicate) {
entry:
  %1 = call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi(ptr addrspace(1) %load_pointer, i1 %predicate, i32 %default_value)
  %2 = call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii(ptr addrspace(1) %load_pointer, i1 %predicate, i32 %default_value, i32 0)
  call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib(ptr addrspace(1) %store_pointer, i32 %store_object, i1 %predicate)
  call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi(ptr addrspace(1) %store_pointer, i32 %store_object, i1 %predicate, i32 0)
  ret void
}

declare spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi(ptr addrspace(1), i1, i32)
declare spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii(ptr addrspace(1), i1, i32, i32)
declare spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib(ptr addrspace(1), i32, i1)
declare spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi(ptr addrspace(1), i32, i1, i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
