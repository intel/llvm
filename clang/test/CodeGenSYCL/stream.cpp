// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s
//
// CHECK: %[[RANGE_TYPE:"struct.*cl::sycl::range"]]
// CHECK: %[[ID_TYPE:"struct.*cl::sycl::id"]]

// CHECK: define dso_local spir_kernel void @{{.*}}StreamTester
// CHECK-SAME: i8 addrspace(1)* [[ACC_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* byval(%[[ID_TYPE]]) align 4 [[ACC_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ACC_INT:%[a-zA-Z0-9_]+]])

// Alloca and addrspace casts for kernel parameters
// CHECK: [[ARG:%[a-zA-Z0-9_]+]].addr = alloca i8 addrspace(1)*, align 8
// CHECK: [[ARG:%[a-zA-Z0-9_]+]].addr.ascast = addrspacecast i8 addrspace(1)** [[ARG]].addr to i8 addrspace(1)* addrspace(4)*
// CHECK: [[ARG_LOAD:%[a-zA-Z0-9_]+]] = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* [[ARG]].addr.ascast, align 8,

// Check __init and __finalize method calls
// CHECK:  call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::stream" addrspace(4)* align 4 dereferenceable_or_null(16) %4, i8 addrspace(1)* [[ARG_LOAD]], %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) {{.*}}%{{.*}}
// CHECK:  call spir_func void @_ZN2cl4sycl6stream10__finalizeEv(%{{.*}}cl::sycl::stream" addrspace(4)* align 4 dereferenceable_or_null(16) %{{[0-9]+}})

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::queue Q;
  Q.submit([&](cl::sycl::handler &CGH) {
    cl::sycl::stream Stream(1024, 128, CGH);

    CGH.single_task<class StreamTester>([=]() {
      Stream << "one" << "two";
    });
  });

  return 0;
}
