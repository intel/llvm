// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv --spirv-ext=+SPV_KHR_subgroup_rotate %t.bc -o %t.spv
// RUN: llvm-spirv --spirv-ext=+SPV_KHR_subgroup_rotate %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-LLVM
// RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPV-IR

// From SPIR-V friendly IR:
// RUN: llvm-spirv %t.rev.bc --spirv-ext=+SPV_KHR_subgroup_rotate -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniformRotateKHR
// CHECK-SPIRV-DAG: Extension "SPV_KHR_subgroup_rotate"

// CHECK-SPIRV-DAG: TypeInt   [[char:[0-9]+]]   8  0
// CHECK-SPIRV-DAG: TypeInt   [[short:[0-9]+]]  16 0
// CHECK-SPIRV-DAG: TypeInt   [[int:[0-9]+]]    32 0
// CHECK-SPIRV-DAG: TypeInt   [[long:[0-9]+]]   64 0
// CHECK-SPIRV-DAG: TypeFloat [[half:[0-9]+]]   16
// CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]]  32
// CHECK-SPIRV-DAG: TypeFloat [[double:[0-9]+]] 64

// CHECK-SPIRV-DAG: Constant [[int]]    [[ScopeSubgroup:[0-9]+]] 3
// CHECK-SPIRV-DAG: Constant [[char]]   [[char_0:[0-9]+]]        0
// CHECK-SPIRV-DAG: Constant [[short]]  [[short_0:[0-9]+]]       0
// CHECK-SPIRV-DAG: Constant [[int]]    [[int_0:[0-9]+]]         0
// CHECK-SPIRV-DAG: Constant [[int]]    [[int_2:[0-9]+]]         2
// CHECK-SPIRV-DAG: Constant [[int]]    [[int_4:[0-9]+]]         4
// CHECK-SPIRV-DAG: Constant [[long]]   [[long_0:[0-9]+]]        0
// CHECK-SPIRV-DAG: Constant [[half]]   [[half_0:[0-9]+]]        0
// CHECK-SPIRV-DAG: Constant [[float]]  [[float_0:[0-9]+]]       0
// CHECK-SPIRV-DAG: Constant [[double]] [[double_0:[0-9]+]]      0

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateChar

// CHECK-LLVM: call spir_func i8 @_Z16sub_group_rotateci(i8 0, i32 2)
// CHECK-LLVM: call spir_func i8 @_Z26sub_group_clustered_rotatecij(i8 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformRotateKHRici(i32 3, i8 0, i32 2)
// CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformRotateKHRicij(i32 3, i8 0, i32 2, i32 4)
kernel void testRotateChar(global char* dst)
{
    char v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateUChar

// CHECK-LLVM: call spir_func i8 @_Z16sub_group_rotateci(i8 0, i32 2)
// CHECK-LLVM: call spir_func i8 @_Z26sub_group_clustered_rotatecij(i8 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformRotateKHRici(i32 3, i8 0, i32 2)
// CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformRotateKHRicij(i32 3, i8 0, i32 2, i32 4)
kernel void testRotateUChar(global uchar* dst)
{
    uchar v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateShort

// CHECK-LLVM: call spir_func i16 @_Z16sub_group_rotatesi(i16 0, i32 2)
// CHECK-LLVM: call spir_func i16 @_Z26sub_group_clustered_rotatesij(i16 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformRotateKHRisi(i32 3, i16 0, i32 2)
// CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformRotateKHRisij(i32 3, i16 0, i32 2, i32 4)
kernel void testRotateShort(global short* dst)
{
    short v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateUShort

// CHECK-LLVM: call spir_func i16 @_Z16sub_group_rotatesi(i16 0, i32 2)
// CHECK-LLVM: call spir_func i16 @_Z26sub_group_clustered_rotatesij(i16 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformRotateKHRisi(i32 3, i16 0, i32 2)
// CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformRotateKHRisij(i32 3, i16 0, i32 2, i32 4)
kernel void testRotateUShort(global ushort* dst)
{
    ushort v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateInt

// CHECK-LLVM: call spir_func i32 @_Z16sub_group_rotateii(i32 0, i32 2)
// CHECK-LLVM: call spir_func i32 @_Z26sub_group_clustered_rotateiij(i32 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformRotateKHRiii(i32 3, i32 0, i32 2)
// CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformRotateKHRiiij(i32 3, i32 0, i32 2, i32 4)
kernel void testRotateInt(global int* dst)
{
    int v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateUInt

// CHECK-LLVM: call spir_func i32 @_Z16sub_group_rotateii(i32 0, i32 2)
// CHECK-LLVM: call spir_func i32 @_Z26sub_group_clustered_rotateiij(i32 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformRotateKHRiii(i32 3, i32 0, i32 2)
// CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformRotateKHRiiij(i32 3, i32 0, i32 2, i32 4)
kernel void testRotateUInt(global uint* dst)
{
    uint v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateLong

// CHECK-LLVM: call spir_func i64 @_Z16sub_group_rotateli(i64 0, i32 2)
// CHECK-LLVM: call spir_func i64 @_Z26sub_group_clustered_rotatelij(i64 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformRotateKHRili(i32 3, i64 0, i32 2)
// CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformRotateKHRilij(i32 3, i64 0, i32 2, i32 4)
kernel void testRotateLong(global long* dst)
{
    long v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateULong

// CHECK-LLVM: call spir_func i64 @_Z16sub_group_rotateli(i64 0, i32 2)
// CHECK-LLVM: call spir_func i64 @_Z26sub_group_clustered_rotatelij(i64 0, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformRotateKHRili(i32 3, i64 0, i32 2)
// CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformRotateKHRilij(i32 3, i64 0, i32 2, i32 4)
kernel void testRotateULong(global ulong* dst)
{
    ulong v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateFloat

// CHECK-LLVM: call spir_func float @_Z16sub_group_rotatefi(float 0.000000e+00, i32 2)
// CHECK-LLVM: call spir_func float @_Z26sub_group_clustered_rotatefij(float 0.000000e+00, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func float @_Z32__spirv_GroupNonUniformRotateKHRifi(i32 3, float 0.000000e+00, i32 2)
// CHECK-SPV-IR: call spir_func float @_Z32__spirv_GroupNonUniformRotateKHRifij(i32 3, float 0.000000e+00, i32 2, i32 4)
kernel void testRotateFloat(global float* dst)
{
    float v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateHalf

// CHECK-LLVM: call spir_func half @_Z16sub_group_rotateDhi(half 0xH0000, i32 2)
// CHECK-LLVM: call spir_func half @_Z26sub_group_clustered_rotateDhij(half 0xH0000, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func half @_Z32__spirv_GroupNonUniformRotateKHRiDhi(i32 3, half 0xH0000, i32 2)
// CHECK-SPV-IR: call spir_func half @_Z32__spirv_GroupNonUniformRotateKHRiDhij(i32 3, half 0xH0000, i32 2, i32 4)
kernel void testRotateHalf(global half* dst)
{
    half v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupNonUniformRotateKHR [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_2]]
// CHECK-SPIRV: GroupNonUniformRotateKHR [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_2]] [[int_4]]
// CHECK-SPIRV: FunctionEnd

// CHECK-COMMON-LABEL: @testRotateDouble

// CHECK-LLVM: call spir_func double @_Z16sub_group_rotatedi(double 0.000000e+00, i32 2)
// CHECK-LLVM: call spir_func double @_Z26sub_group_clustered_rotatedij(double 0.000000e+00, i32 2, i32 4)

// CHECK-SPV-IR: call spir_func double @_Z32__spirv_GroupNonUniformRotateKHRidi(i32 3, double 0.000000e+00, i32 2)
// CHECK-SPV-IR: call spir_func double @_Z32__spirv_GroupNonUniformRotateKHRidij(i32 3, double 0.000000e+00, i32 2, i32 4)
kernel void testRotateDouble(global double* dst)
{
    double v = 0;
    dst[0] = sub_group_rotate(v, 2);
    dst[1] = sub_group_clustered_rotate(v, 2, 4);
}
