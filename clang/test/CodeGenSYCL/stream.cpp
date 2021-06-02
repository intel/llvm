// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s
//
// CHECK: define {{.*}}spir_kernel void @"{{.*}}StreamTester"(%"{{.*}}cl::sycl::stream"* byval(%"{{.*}}cl::sycl::stream") {{.*}}){{.*}}
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::stream{{.*}} addrspace(4)* {{[^,]*}} %{{[0-9]+}})
// CHECK: call spir_func void @{{.*}}__finalize{{.*}}(%{{.*}}cl::sycl::stream{{.*}} addrspace(4)* {{[^,]*}} %{{[0-9]+}})
//

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
