// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

float acos(float val) {
  // CHECK: @_Z4acosf
  // CHECK: call noundef float @_Z16__spirv_ocl_acosf
  return __spirv_ocl_acos(val);
}

// CHECK: declare noundef float @_Z16__spirv_ocl_acosf(float noundef)

double acos(double val) {
  // CHECK: @_Z4acosd
  // CHECK: call noundef double @_Z16__spirv_ocl_acosd
  return __spirv_ocl_acos(val);
}

// CHECK: declare noundef double @_Z16__spirv_ocl_acosd(double noundef)

void control_barrier() {
  // CHECK-LABEL: @_Z15control_barrierv
  // CHECK: call void @_Z22__spirv_ControlBarrieriii
  __spirv_ControlBarrier(2, 2, 912);
}

// CHECK: declare void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)

void memory_barrier() {
  // CHECK-LABEL: @_Z14memory_barrierv
  // CHECK: call void @_Z21__spirv_MemoryBarrierii(
  __spirv_MemoryBarrier(0, 2);
}

// CHECK: declare void @_Z21__spirv_MemoryBarrierii(i32 noundef, i32 noundef)
