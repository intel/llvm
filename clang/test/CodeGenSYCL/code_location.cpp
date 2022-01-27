// RUN: %clang_cc1 -fsycl-is-device -internal-isystem -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task<class KernelName>([]() {}); });
  return 0;
}

// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '4', 'm', 'a', 'i', 'n', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '2', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK: static constexpr const char* getFileName() { return {{.*}}/clang/test/CodeGenSYCL/code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return ; }
// CHECK: static constexpr unsigned getLineNumber() { return 8; }
// CHECK: static constexpr unsigned getColumnNumber() { return 54; }
//};
// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK: static constexpr const char* getFileName() { return {{.*}}/clang/test/CodeGenSYCL/code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return KernelName; }
// CHECK: static constexpr unsigned getLineNumber() { return 9; }
// CHECK: static constexpr unsigned getColumnNumber() { return 72; }
//};
