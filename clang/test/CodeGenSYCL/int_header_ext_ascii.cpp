// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"

// Test ensures that we can properly handle unnamed kernels with extended-ascii
// characters in it.

struct ΩßPolicyßΩ {
  void operator()() const {}
};

void use() {
  ΩßPolicyßΩ Functor;

  sycl::handler h;
  h.single_task(Functor);
}

// CHECK: // Forward declarations of templated kernel function types:
// CHECK: struct KernelInfoData<'_', 'Z', 'T', 'S', '1', '4', -50, -87, -61, -97, 'P', 'o', 'l', 'i', 'c', 'y', -61, -97, -50, -87> {
// CHECK: static constexpr const char* getName() { return "_ZTS14ΩßPolicyßΩ"; }
