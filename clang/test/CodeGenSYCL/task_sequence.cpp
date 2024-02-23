// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Test that SPIR-V codegen generates the expected LLVM struct name for the
// TaskSequenceINTEL type.

#include <stddef.h>
#include <stdint.h>

namespace __spv {
  template <typename T>
  struct __spirv_TaskSequenceINTEL;
}

struct S {
  char c;
  float f;
};

// CHECK: @_Z2f1{{.*}}(target("spirv.TaskSequenceINTEL", float)
void f1(__spv::__spirv_TaskSequenceINTEL<float> *task_seq) {}

// CHECK: @_Z2f2{{.*}}(target("spirv.TaskSequenceINTEL", i64)
void f2(__spv::__spirv_TaskSequenceINTEL<uint64_t> *task_seq) {}

// CHECK: @_Z2f3{{.*}}(target("spirv.TaskSequenceINTEL", i8)
void f3(__spv::__spirv_TaskSequenceINTEL<char> *task_seq) {}

// CHECK: @_Z2f4{{.*}}(target("spirv.TaskSequenceINTEL", i128)
void f4(__spv::__spirv_TaskSequenceINTEL<_BitInt(128)> *task_seq) {}

// CHECK: @_Z2f5{{.*}}(target("spirv.TaskSequenceINTEL", double)
void f5(__spv::__spirv_TaskSequenceINTEL<double> *task_seq) {}

// CHECK: @_Z2f6{{.*}}(target("spirv.TaskSequenceINTEL", %struct.S = type { i8, float })
void f6(__spv::__spirv_TaskSequenceINTEL<S> *task_seq) {}
