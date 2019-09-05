// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx --sycl -S -x c++ -O0 -emit-llvm %s -o - | FileCheck %s
#include <CL/sycl.hpp>
using namespace cl::sycl;

template <typename T>
class foo;

template<typename T>
void kernel_func(T val) {
  cl::sycl::queue testQueue;

  T data = val;
  buffer<T,1> buf(&data, range<1>(1));

  testQueue.submit([&](cl::sycl::handler &cgh) {
    auto GlobAcc = buf.template get_access<access::mode::atomic>(cgh);
    cgh.single_task<class foo<T>>([=]() {
      auto a = GlobAcc[0];
      T var = a.load();
    });
  });
}

int main() {
  // CHECK: [[U_NAME:%[0-9a-zA-Z]*]] = alloca %union
  // CHECK: call spir_func i32 @_Z18__spirv_AtomicLoad{{.*}}(i32 addrspace(1)* %{{[0-9a-zA-Z]*}}, i32 1, i32 %{{[0-9a-zA-Z]*}})
  // CHECK: bitcast %union{{.*}} [[U_NAME]] to i32*
  // CHECK: [[F_NAME:%[0-9a-zA-Z]*]] = bitcast %union{{.*}} [[U_NAME]] to float*
  // CHECK: [[RET_NAME:%[0-9a-zA-Z]*]] = {{.*}}[[F_NAME]]
  // CHECK: ret float [[RET_NAME]]
  kernel_func<float>(5.5);
  // CHECK: [[RET_NAME2:%[0-9a-zA-Z]*]] = call spir_func i32 @_Z18__spirv_AtomicLoad{{.*}}(i32 addrspace(1)* %{{[0-9a-zA-Z]*}}, i32 1, i32 %{{[0-9a-zA-Z]*}})
  // CHECK: ret i32 [[RET_NAME2]]
  kernel_func<int>(42);
  return 0;
}