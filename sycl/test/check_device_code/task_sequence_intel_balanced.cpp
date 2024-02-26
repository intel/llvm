// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

// CHECK: [[TASK_SEQUENCE:%.*]] = call spir_func target("spirv.TaskSequenceINTEL") @_Z31__spirv_TaskSequenceCreateINTEL{{.*}}(ptr {{.*}} @_Z8user_sot{{.*}}, i32 -1, i32 -1, i32 2, i32 2)
// CHECK: call spir_func void @_Z30__spirv_TaskSequenceAsyncINTEL{{.*}}(target("spirv.TaskSequenceINTEL") [[TASK_SEQUENCE]], ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, i32 128)
// CHECK-COUNT-1: call spir_func i32 @_Z28__spirv_TaskSequenceGetINTEL{{.*}}(target("spirv.TaskSequenceINTEL")[[TASK_SEQUENCE]])
// CHECK: call spir_func void @_Z32__spirv_TaskSequenceReleaseINTEL{{.*}}(target("spirv.TaskSequenceINTEL")[[TASK_SEQUENCE]])

#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::oneapi::experimental;

constexpr int NSIZE = 128;

int arrayAdd(int *data1, int *data2, int N) {
  int ret = 0;
  for (int i = 0; i < N; ++i) {
    ret += data1[i] + data2[i];
  }

  return ret;
}

int main() {
  sycl::queue myQueue;
  int result = 0;
  myQueue.submit([&](sycl::handler &cgh) {
    sycl::buffer<int, 1> result_sycl(&result, sycl::range<1>(1));
    auto result_acc = result_sycl.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=](sycl::kernel_handler kh) {
      int d1[NSIZE], d2[NSIZE];
      task_sequence<arrayAdd,
                    decltype(properties{balanced, invocation_capacity<2>,
                                        response_capacity<2>})>
          sot_object;
      sot_object.async(d1, d2, NSIZE);
      result_acc[0] = sot_object.get();
    });
  });
  myQueue.wait();
  return result;
}