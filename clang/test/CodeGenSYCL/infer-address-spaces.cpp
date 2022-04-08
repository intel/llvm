// RUN:  %clang_cc1 -O1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

#define BLOCK_SIZE 16

using namespace cl::sycl;

int main() {
  queue Q;
  auto MatrixTemp = buffer<float, 1>{range<1>{BLOCK_SIZE * 512}};
  Q.submit([&](handler &cgh) {
    auto temp_dst_acc = MatrixTemp.get_access<access::mode::write>(cgh);
    auto temp_t = accessor<float, 1, access::mode::read_write, access::target::local>();
    cgh.parallel_for<class test>(range<1>(BLOCK_SIZE), [=](id<1> id) {
      int index = 64 * id.get(0);
      temp_dst_acc[index] = temp_t[index];
    });
  });

  return 0;
}

// No addrspacecast before loading and storing values
// CHECK: %[[#VALUE_1:]] = getelementptr inbounds %"struct.cl::sycl::range", %"struct.cl::sycl::range"* %{{.*}}, i64 0, i32 0
// CHECK-NOT: %{{.*}} = addrspacecast i32* %[[#VALUE_1]] to i32 addrspace(4)*
// CHECK: %[[#VALUE_2:]] = getelementptr inbounds %"struct.cl::sycl::range", %"struct.cl::sycl::range"* %{{.*}}, i64 0, i32 0
// CHECK-NOT: %{{.*}} = addrspacecast i32* %[[#VALUE_2]] to i32 addrspace(4)*
// CHECK-NOT: %{{.*}} = load i32, i32 addrspace(4)* %[[#VALUE_1]], align 4, !tbaa !6
// CHECK: %[[#VALUE_3:]] = load i32, i32* %[[#VALUE_1]], align 4, !tbaa !6
// CHECK-NOT: store i32 %[[#VALUE_3]], i32 addrspace(4)* %[[#VALUE_2]], align 4, !tbaa !6
// CHECK: store i32 %[[#VALUE_3]], i32* %[[#VALUE_2]], align 4, !tbaa !6
