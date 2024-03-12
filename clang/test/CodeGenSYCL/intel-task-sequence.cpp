// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Test that SPIR-V codegen generates the expected LLVM struct name for the
// TaskSequenceINTEL type

namespace __spv {
  struct __spirv_TaskSequenceINTEL;
} // namespace __spv

// CHECK: @_Z4func{{.*}}(target("spirv.TaskSequenceINTEL")
void func(__spv::__spirv_TaskSequenceINTEL *task_seq) {}