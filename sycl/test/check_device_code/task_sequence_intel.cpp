// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

// CHECK: [[TaskSequence:%.*task_sequence"]] = type { i32, i64 }
// CHECK: call spir_func i64 @_Z31__spirv_TaskSequenceCreateINTEL{{.*}}([[TaskSequence]] addrspace(4)*{{.*}}, i32 (i32 addrspace(4)*, i32 addrspace(4)*, i32)*{{.*}}[[Function:@.*]], i32 0)
// CHECK: call spir_func void @_Z30__spirv_TaskSequenceAsyncINTEL{{.*}}([[TaskSequence]] addrspace(4)*{{.*}}, i32 (i32 addrspace(4)*, i32 addrspace(4)*, i32)*{{.*}}[[Function]], i64{{.*}}, i32 2, i32 addrspace(4)*{{.*}}, i32 addrspace(4)*{{.*}}, i32 128)
// CHECK: call spir_func i32 @_Z28__spirv_TaskSequenceGetINTEL{{.*}}([[TaskSequence]] addrspace(4)*{{.*}}, i32 (i32 addrspace(4)*, i32 addrspace(4)*, i32)*{{.*}}[[Function]], i64{{.*}}, i32 2)
// CHECK: call spir_func void @_Z32__spirv_TaskSequenceReleaseINTEL{{.*}}([[TaskSequence]] addrspace(4)*{{.*}})

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>

using sycl::ext::intel::experimental::task_sequence;

constexpr int NSIZE = 128;

int UserTask(int* data1, int* data2, int N) {
  int ret = 0;
  for (int i = 0; i < N; ++i)
    ret += data1[i] + data2[i];
  return ret;
}

int main () {
  sycl::queue myQueue;
  myQueue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task([=](cl::sycl::kernel_handler kh) {
      int d1[NSIZE], d2[NSIZE];
      task_sequence<UserTask, 2, 2> taskObj;
      taskObj.async(d1, d2, NSIZE);
    });
  });
  myQueue.wait();
}
