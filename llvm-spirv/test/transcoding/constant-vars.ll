; Check that we can handle constant expressions correctly.
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-DAG: 4 TypeInt [[U8:[0-9]+]] 8 0
; CHECK-SPIRV-DAG: 4 TypeInt [[U32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: 4 TypeInt [[U64:[0-9]+]] 64 0
; CHECK-SPIRV-DAG: 4 Constant [[U32]] [[I320:[0-9]+]] 0
; CHECK-SPIRV-DAG: 4 Constant [[U32]] [[I323:[0-9]+]] 3
; CHECK-SPIRV: 4 TypePointer [[AS2:[0-9]+]] 0 [[U8]]
; CHECK-SPIRV: 4 TypePointer [[AS1:[0-9]+]] 5 [[U8]]
; CHECK-SPIRV-DAG: 4 TypeStruct [[STRUCTTY:[0-9]+]]
; CHECK-SPIRV-DAG: 4 TypeArray [[ARRAYTY:[0-9]+]] [[AS1]] [[I323]]

@astr = internal addrspace(2) constant [7 x i8] c"string\00", align 4
; CHECK-SPIRV: 5 Variable {{[0-9]+}} [[ASTR:[0-9]+]] 0

@i64arr = addrspace(1) constant [3 x i64] [i64 0, i64 1, i64 2]
; CHECK-SPIRV: 5 Variable {{[0-9]+}} [[I64ARR:[0-9]+]] 5

@struct = addrspace(1) global {ptr addrspace(2), ptr addrspace(1)} { ptr addrspace(2) @astr, ptr addrspace(1) @i64arr }

; CHECK-SPIRV: 7 SpecConstantOp [[AS2]] [[ASTRC:[0-9]+]] 70 [[ASTR]] [[I320]] [[I320]]
; CHECK-SPIRV: 5 SpecConstantOp [[AS1]] [[I64ARRC:[0-9]+]] 124 [[I64ARR]]
; CHECK-SPIRV: 5 ConstantComposite [[STRUCTTY]] [[STRUCT_INIT:[0-9]+]] [[ASTRC]] [[I64ARRC]]
; CHECK-SPIRV: 5 Variable {{[0-9]+}} [[STRUCT:[0-9]+]] 5 [[STRUCT_INIT]]

@array = addrspace(1) global [3 x ptr addrspace(1)] [ptr addrspace(1) @i64arr, ptr addrspace(1) @struct, ptr addrspace(1) getelementptr ([3 x i64], ptr addrspace(1) @i64arr, i64 0, i64 1) ]

; CHECK-SPIRV: 5 SpecConstantOp [[AS1]] [[I64ARRC2:[0-9]+]] 124 [[I64ARR]]
; CHECK-SPIRV: 5 SpecConstantOp [[AS1]] [[STRUCTC:[0-9]+]] 124 [[STRUCT]]
; CHECK-SPIRV: 7 SpecConstantOp {{[0-9]+}} [[GEP:[0-9]+]] 67 [[I64ARR]]
; CHECK-SPIRV: 6 ConstantComposite [[ARRAYTY]] [[ARRAY_INIT:[0-9]+]] [[I64ARRC2]] [[STRUCTC]] [[GEP]]
; CHECK-SPIRV: 5 Variable {{[0-9]+}} [[ARRAY:[0-9]+]] 5 [[ARRAY_INIT]]

; CHECK-LLVM: %structtype = type { ptr addrspace(2), ptr addrspace(1) }
; CHECK-LLVM: @astr = internal unnamed_addr addrspace(2) constant [7 x i8] c"string\00", align 4
; CHECK-LLVM: @i64arr = addrspace(1) constant [3 x i64] [i64 0, i64 1, i64 2]
; CHECK-LLVM: @struct = addrspace(1) global %structtype { ptr addrspace(2) @astr, ptr addrspace(1) @i64arr }
; CHECK-LLVM: @array = addrspace(1) global [3 x ptr addrspace(1)] [ptr addrspace(1) @i64arr, ptr addrspace(1) @struct, ptr addrspace(1) getelementptr ([3 x i64], ptr addrspace(1) @i64arr, i64 0, i64 1)]

define spir_kernel void @foo() {
  %val = load i32, ptr addrspace(2) @astr, align 4
  ret void
}
