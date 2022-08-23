// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s
//
// CHECK: %[[RANGE_TYPE:"struct.*sycl::_V1::range"]]
// CHECK: %[[ID_TYPE:"struct.*sycl::_V1::id"]]

// CHECK: define dso_local spir_kernel void @{{.*}}StreamTester
// CHECK-SAME: ptr addrspace(1) noundef align 1 [[ACC_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[ID_TYPE]]) align 4 [[ACC_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 noundef [[ACC_INT:%[a-zA-Z0-9_]+]])

// Alloca and addrspace casts for kernel parameters
// CHECK: [[ARG:%[a-zA-Z0-9_]+]].addr = alloca ptr addrspace(1), align 8
// CHECK: [[ARG:%[a-zA-Z0-9_]+]].addr.ascast = addrspacecast ptr [[ARG]].addr to ptr addrspace(4)
// CHECK: [[ARG_LOAD:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[ARG]].addr.ascast, align 8,

// Check __init and __finalize method calls
// CHECK:  call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %{{[a-zA-Z0-9_]+}}, ptr addrspace(1) noundef [[ARG_LOAD]], ptr noundef byval(%[[RANGE_TYPE]]) {{.*}}%{{.*}}
// CHECK:  call spir_func void @_ZN4sycl3_V16stream10__finalizeEv(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %{{[a-zA-Z0-9_]+}})

#include "Inputs/sycl.hpp"

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream Stream(1024, 128, CGH);

    CGH.single_task<class StreamTester>([=]() {
      Stream << "one" << "two";
    });
  });

  return 0;
}
