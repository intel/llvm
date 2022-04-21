// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR

// CHECK-SPIRV: SatConvertSToU

// CHECK-LLVM-LABEL: @testSToU
// CHECK-LLVM: call spir_func <2 x i8> @_Z18convert_uchar2_satDv2_i

// CHECK-SPV-IR-LABEL: @testSToU
// CHECK-SPV-IR: call spir_func <2 x i8> @_Z30__spirv_SatConvertSToU_Ruchar2Dv2_i

kernel void testSToU(global int2 *a, global uchar2 *res) {
  res[0] = convert_uchar2_sat(*a);
}

// CHECK-SPIRV: SatConvertUToS

// CHECK-LLVM-LABEL: @testUToS
// CHECK-LLVM: call spir_func <2 x i8> @_Z17convert_char2_satDv2_j

// CHECK-SPV-IR-LABEL: @testUToS
// CHECK-SPV-IR: call spir_func <2 x i8> @_Z29__spirv_SatConvertUToS_Rchar2Dv2_j
kernel void testUToS(global uint2 *a, global char2 *res) {
  res[0] = convert_char2_sat(*a);
}

// CHECK-SPIRV: ConvertUToF

// CHECK-LLVM-LABEL: @testUToF
// CHECK-LLVM: call spir_func <2 x float> @_Z18convert_float2_rtzDv2_j

// CHECK-SPV-IR-LABEL: @testUToF
// CHECK-SPV-IR: call spir_func <2 x float> @_Z31__spirv_ConvertUToF_Rfloat2_rtzDv2_j
kernel void testUToF(global uint2 *a, global float2 *res) {
  res[0] = convert_float2_rtz(*a);
}

// CHECK-SPIRV: ConvertFToU

// CHECK-LLVM-LABEL: @testFToUSat
// CHECK-LLVM: call spir_func <2 x i32> @_Z21convert_uint2_sat_rtnDv2_f

// CHECK-SPV-IR-LABEL: @testFToUSat
// CHECK-SPV-IR: call spir_func <2 x i32> @_Z34__spirv_ConvertFToU_Ruint2_sat_rtnDv2_f
kernel void testFToUSat(global float2 *a, global uint2 *res) {
  res[0] = convert_uint2_sat_rtn(*a);
}

// CHECK-SPIRV: UConvert

// CHECK-LLVM-LABEL: @testUToUSat
// CHECK-LLVM: call spir_func i32 @_Z16convert_uint_sath

// CHECK-SPV-IR-LABEL: @testUToUSat
// CHECK-SPV-IR: call spir_func i32 @_Z26__spirv_UConvert_Ruint_sath
kernel void testUToUSat(global uchar *a, global uint *res) {
  res[0] = convert_uint_sat(*a);
}

// CHECK-SPIRV: UConvert

// CHECK-LLVM-LABEL: @testUToUSat1
// CHECK-LLVM: call spir_func i8 @_Z17convert_uchar_satj

// CHECK-SPV-IR-LABEL: @testUToUSat1
// CHECK-SPV-IR: call spir_func i8 @_Z27__spirv_UConvert_Ruchar_satj
kernel void testUToUSat1(global uint *a, global uchar *res) {
  res[0] = convert_uchar_sat(*a);
}

// CHECK-SPIRV: ConvertFToU

// CHECK-LLVM-LABEL: @testFToU
// CHECK-LLVM: call spir_func <3 x i32> @_Z17convert_uint3_rtpDv3_f

// CHECK-SPV-IR-LABEL: @testFToU
// CHECK-SPV-IR: call spir_func <3 x i32> @_Z30__spirv_ConvertFToU_Ruint3_rtpDv3_f
kernel void testFToU(global float3 *a, global uint3 *res) {
  res[0] = convert_uint3_rtp(*a);
}
