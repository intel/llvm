// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "sycl.hpp"

// Test that SPIR-V codegen generates the expected LLVM struct name for the
// TaskSequenceINTEL type, and the expected functions (arg and return types)

using namespace sycl;
queue q;

void task_function() {}

// CHECK: declare {{.*}} target("spirv.TaskSequenceINTEL") @_Z31__spirv_TaskSequenceCreateINTEL{{.*}}(ptr noundef, i32 noundef, i16 noundef zeroext, i32 noundef, i32 noundef)
// CHECK: declare {{.*}} void @_Z30__spirv_TaskSequenceAsyncINTEL{{.*}}(target("spirv.TaskSequenceINTEL") noundef)
// CHECK: declare {{.*}} void @_Z28__spirv_TaskSequenceGetINTEL{{.*}}(target("spirv.TaskSequenceINTEL") noundef)
// CHECK: declare {{.*}} void @_Z32__spirv_TaskSequenceReleaseINTEL{{.*}}(target("spirv.TaskSequenceINTEL") noundef)

int main() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name>([]() {
      ext::intel::experimental::task_sequence<task_function> fake_task;
      fake_task.async();
      fake_task.get();
    });
  });
  return 0;
}
