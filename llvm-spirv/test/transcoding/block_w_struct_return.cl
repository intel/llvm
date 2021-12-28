// RUN: %clang_cc1 -triple spir -cl-std=cl2.0 -disable-llvm-passes -fdeclare-opencl-builtins -finclude-default-header %s -emit-llvm-bc -o %t.bc

// RUN: llvm-spirv --spirv-max-version=1.1 %t.bc -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV1_1,CHECK-SPIRV
// RUN: llvm-spirv --spirv-max-version=1.1 %t.bc -o %t.spirv1.1.spv
// RUN: spirv-val --target-env spv1.1 %t.spirv1.1.spv
// RUN: llvm-spirv -r %t.spirv1.1.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc
// RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

// RUN: llvm-spirv --spirv-max-version=1.4 %t.bc -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV1_4,CHECK-SPIRV
// RUN: llvm-spirv --spirv-max-version=1.4 %t.bc -o %t.spirv1.4.spv
// RUN: spirv-val --target-env spv1.4 %t.spirv1.4.spv
// RUN: llvm-spirv -r %t.spirv1.4.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc
// RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

kernel void block_ret_struct(__global int* res)
{
  struct A {
      int a;
  };
  struct A (^kernelBlock)(struct A) = ^struct A(struct A a)
  {
    a.a = 6;
    return a;
  };
  size_t tid = get_global_id(0);
  res[tid] = -1;
  struct A aa;
  aa.a = 5;
  res[tid] = kernelBlock(aa).a - 6;
}

// CHECK-SPIRV1_4: EntryPoint 6 [[#]] "block_ret_struct" [[#InterdaceId1:]] [[#InterdaceId2:]]
// CHECK-SPIRV1_4: Name [[#InterdaceId1]] "__block_literal_global"
// CHECK-SPIRV1_4: Name [[#InterdaceId2]] "__spirv_BuiltInGlobalInvocationId"

// CHECK-SPIRV1_1: EntryPoint 6 [[#]] "block_ret_struct" [[#InterdaceId1:]]
// CHECK-SPIRV1_1: Name [[#InterdaceId1]] "__spirv_BuiltInGlobalInvocationId"

// CHECK-SPIRV: Name [[BlockInv:[0-9]+]] "__block_ret_struct_block_invoke"

// CHECK-SPIRV: 4 TypeInt [[IntTy:[0-9]+]] 32
// CHECK-SPIRV: 4 TypeInt [[Int8Ty:[0-9]+]] 8
// CHECK-SPIRV: 4 TypePointer [[Int8Ptr:[0-9]+]] 8 [[Int8Ty]]
// CHECK-SPIRV: 3 TypeStruct [[StructTy:[0-9]+]] [[IntTy]]
// CHECK-SPIRV: 4 TypePointer [[StructPtrTy:[0-9]+]] 7 [[StructTy]]

// CHECK-SPIRV: 4 Variable [[StructPtrTy]] [[StructArg:[0-9]+]] 7
// CHECK-SPIRV: 4 Variable [[StructPtrTy]] [[StructRet:[0-9]+]] 7
// CHECK-SPIRV: 4 PtrCastToGeneric [[Int8Ptr]] [[BlockLit:[0-9]+]] {{[0-9]+}}
// CHECK-SPIRV: 7 FunctionCall {{[0-9]+}} {{[0-9]+}} [[BlockInv]] [[StructRet]] [[BlockLit]] [[StructArg]]

// CHECK-LLVM: %[[StructA:.*]] = type { i32 }
// CHECK-LLVM: call {{.*}} void @__block_ret_struct_block_invoke(%[[StructA]]*
