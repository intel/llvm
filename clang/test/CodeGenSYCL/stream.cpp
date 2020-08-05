// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -I %S/Inputs -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s
//
// CHECK: %[[RANGE_TYPE:"struct.*cl::sycl::range"]]
// CHECK: %[[ID_TYPE:"struct.*cl::sycl::id"]]
// CHECK: define spir_kernel void @{{.*}}StreamTester
// CHECK-SAME: i8 addrspace(1)* [[ACC_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* byval(%[[ID_TYPE]]) align 4 [[ACC_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ARG_INT:%[a-zA-Z0-9_]+]])

// CHECK: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::stream{{.*}} addrspace(4)* %{{[0-9]+}}, i8 addrspace(1)* %5, %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) {{.*}} %{{.*}}
// CHECK: call spir_func void @{{.*}}__finalize{{.*}}(%{{.*}}cl::sycl::stream{{.*}} addrspace(4)* %{{[0-9]+}})
//

#include "sycl.hpp"

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
