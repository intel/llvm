; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@.str = internal unnamed_addr addrspace(2) constant [11 x i8] c"Value: %p\0A\00", align 1


; CHECK-DAG: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK-DAG: 4 TypeInt [[CHAR:[0-9]+]] 8 0
; CHECK-DAG: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]
; CHECK-DAG: 4 TypePointer [[CHARPTR:[0-9]+]] 0 [[CHAR]]
; CHECK: 5 Variable {{[0-9]+}} [[STR:[0-9]+]] 0
; CHECK: 4 Variable [[INTPTR]] [[IPTR:[0-9]+]] 7
; CHECK: 4 Bitcast [[CHARPTR]] [[I8STR:[0-9]+]] [[STR]]
; CHECK: 7 ExtInst [[INT]] {{[0-9]+}} {{[0-9]+}} printf [[I8STR]] [[IPTR]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
  %iptr = alloca i32, align 4
  %res = call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2cz(ptr addrspace(2) @.str, ptr %iptr)
  ret void
}

declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS2cz(ptr addrspace(2), ...)
