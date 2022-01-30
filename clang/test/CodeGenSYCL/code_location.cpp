// RUN: %clang_cc1 -fsycl-is-device -internal-isystem -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem -sycl-std=2020 -DNDEBUG -fsycl-int-header=%t2.h %s -o %t2.out
// RUN: FileCheck -input-file=%t2.h %s

#include "Inputs/sycl.hpp"
#ifndef NDEBUG
int test1() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task<class KernelName>([]() {}); });
  return 0;
}
// Specializations of KernelInfo for kernel function types:
// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '5', 't', 'e', 's', 't', '1', 'v', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '2', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return ; }
// CHECK: static constexpr unsigned getLineNumber() { return 10; }
// CHECK: static constexpr unsigned getColumnNumber() { return 54; }
//};

// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return KernelName; }
// CHECK: static constexpr unsigned getLineNumber() { return 11; }
// CHECK: static constexpr unsigned getColumnNumber() { return 72; }
//};

class KernelName2;
int test2() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task(
                                           [] { int i = 2; }); });
  q.submit([&](cl::sycl::handler &h) { h.single_task<KernelName2>(
                                           [] { int i = 2; }); });
  return 0;
}

// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '5', 't', 'e', 's', 't', '2', 'v', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '2', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return ; }
// CHECK: static constexpr unsigned getLineNumber() { return 33; }
// CHECK: static constexpr unsigned getColumnNumber() { return 44; }
//};
// CHECK: template <> struct KernelInfo<::KernelName2> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return KernelName2; }
// CHECK: static constexpr unsigned getLineNumber() { return 35; }
// CHECK: static constexpr unsigned getColumnNumber() { return 44; }
//};

template <typename T> class KernelName3;
int test3() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task<KernelName3<KernelName2>>(
                                           [] { int i = 3; }); });
  return 0;
}

// CHECK: template <> struct KernelInfo<::KernelName3<::KernelName2>> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return KernelName3; }
// CHECK: static constexpr unsigned getLineNumber() { return 56; }
// CHECK: static constexpr unsigned getColumnNumber() { return 44; }
//};

auto l4 = []() { return 4; };
int test4() {
  cl::sycl::queue q;
  q.submit([=](cl::sycl::handler &h) { h.single_task<class KernelName4>(l4); });
  return 0;
}

// CHECK: template <> struct KernelInfo<KernelName4> {
// CHECK: static constexpr const char* getFileName() { return code_location.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return KernelName4; }
// CHECK: static constexpr unsigned getLineNumber() { return 67; }
// CHECK: static constexpr unsigned getColumnNumber() { return 11; }
//};

#else
int test5() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task<class KernelName5>([]() {}); });
  return 0;
}

// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '5', 't', 'e', 's', 't', '5', 'v', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '2', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK: static constexpr const char* getFileName() { return ; }
// CHECK: static constexpr const char* getFunctionName() { return ; }
// CHECK: static constexpr unsigned getLineNumber() { return 0; }
// CHECK: static constexpr unsigned getColumnNumber() { return 0; }
//};
// CHECK: template <> struct KernelInfo<KernelName5> {
// CHECK: static constexpr const char* getFileName() { return ; }
// CHECK: static constexpr const char* getFunctionName() { return ; }
// CHECK: static constexpr unsigned getLineNumber() { return 0; }
// CHECK: static constexpr unsigned getColumnNumber() { return 0; }
//};
#endif
