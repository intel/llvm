// REQUIRES: spirv-dis
// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_shader_clock -o %t.spv
// RUN: spirv-dis %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// TODO: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR

// CHECK-SPIRV: OpCapability ShaderClockKHR
// CHECK-SPIRV: OpExtension "SPV_KHR_shader_clock"
// CHECK-SPIRV-DAG: [[uint:%[a-z0-9_]+]] = OpTypeInt 32
// CHECK-SPIRV-DAG: [[ulong:%[a-z0-9_]+]] = OpTypeInt 64
// CHECK-SPIRV-DAG: [[v2uint:%[a-z0-9_]+]] = OpTypeVector [[uint]] 2
// CHECK-SPIRV-DAG: [[uint_1:%[a-z0-9_]+]] = OpConstant [[uint]] 1
// CHECK-SPIRV-DAG: [[uint_2:%[a-z0-9_]+]] = OpConstant [[uint]] 2
// CHECK-SPIRV-DAG: [[uint_3:%[a-z0-9_]+]] = OpConstant [[uint]] 3
// CHECK-SPIRV: OpReadClockKHR [[ulong]] [[uint_1]]
// CHECK-SPIRV: OpReadClockKHR [[ulong]] [[uint_2]]
// CHECK-SPIRV: OpReadClockKHR [[ulong]] [[uint_3]]
// CHECK-SPIRV: OpReadClockKHR [[v2uint]] [[uint_1]]
// CHECK-SPIRV: OpReadClockKHR [[v2uint]] [[uint_2]]
// CHECK-SPIRV: OpReadClockKHR [[v2uint]] [[uint_3]]

// CHECK-LLVM-LABEL: test_clocks
// CHECK-LLVM: call spir_func i64 @_Z17clock_read_devicev()
// CHECK-LLVM: call spir_func i64 @_Z21clock_read_work_groupv()
// CHECK-LLVM: call spir_func i64 @_Z20clock_read_sub_groupv()
// CHECK-LLVM: call spir_func <2 x i32> @_Z22clock_read_hilo_devicev()
// CHECK-LLVM: call spir_func <2 x i32> @_Z26clock_read_hilo_work_groupv()
// CHECK-LLVM: call spir_func <2 x i32> @_Z25clock_read_hilo_sub_groupv()

// CHECK-SPV-IR-LABEL: test_clocks
// CHECK-SPV-IR: call spir_func i64 @_Z27__spirv_ReadClockKHR_Rulongi(i32 1)
// CHECK-SPV-IR: call spir_func i64 @_Z27__spirv_ReadClockKHR_Rulongi(i32 2)
// CHECK-SPV-IR: call spir_func i64 @_Z27__spirv_ReadClockKHR_Rulongi(i32 3)
// CHECK-SPV-IR: call spir_func <2 x i32> @_Z27__spirv_ReadClockKHR_Ruint2i(i32 1)
// CHECK-SPV-IR: call spir_func <2 x i32> @_Z27__spirv_ReadClockKHR_Ruint2i(i32 2)
// CHECK-SPV-IR: call spir_func <2 x i32> @_Z27__spirv_ReadClockKHR_Ruint2i(i32 3)

kernel void test_clocks(global ulong *out64, global uint2 *outv2) {
  out64[0] = clock_read_device();
  out64[1] = clock_read_work_group();
  out64[2] = clock_read_sub_group();

  outv2[0] = clock_read_hilo_device();
  outv2[1] = clock_read_hilo_work_group();
  outv2[2] = clock_read_hilo_sub_group();
}
