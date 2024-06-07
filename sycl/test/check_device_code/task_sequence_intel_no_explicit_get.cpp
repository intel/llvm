// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

// CHECK: [[TASK_SEQUENCE:%.*]] ={{.*}} call spir_func target("spirv.TaskSequenceINTEL") @_Z31__spirv_TaskSequenceCreateINTEL{{.*}}(ptr{{.*}}@_Z8arrayAdd{{.*}}, i32 0, i32 1, i32 1, i32 1)
// CHECK: call spir_func void @_Z30__spirv_TaskSequenceAsyncINTEL{{.*}}(target("spirv.TaskSequenceINTEL") [[TASK_SEQUENCE]], ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, i32 128)
// CHECK-COUNT-1: call spir_func i32 @_Z28__spirv_TaskSequenceGetINTEL{{.*}}(target("spirv.TaskSequenceINTEL") [[TASK_SEQUENCE]])
// CHECK: call spir_func void @_Z32__spirv_TaskSequenceReleaseINTEL{{.*}}(target("spirv.TaskSequenceINTEL") [[TASK_SEQUENCE]])

#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::oneapi::experimental;

constexpr int kSize = 128;

int arrayAdd(int *data1, int *data2, int N) {
  int ret = 0;
  for (int i = 0; i < N; ++i) {
    ret += data1[i] + data2[i];
  }

  return ret;
}

SYCL_EXTERNAL void task_sequence_no_explicit_get() {
  int d1[kSize], d2[kSize];
  task_sequence<arrayAdd, decltype(properties{
                              pipelined<0>, stall_enable_clusters,
                              invocation_capacity<1>, response_capacity<1>})>
      arrayAddTask;
  arrayAddTask.async(d1, d2, kSize);
}