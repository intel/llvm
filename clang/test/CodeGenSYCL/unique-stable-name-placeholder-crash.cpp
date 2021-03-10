// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -internal-isystem %S/Inputs -std=c++17 -sycl-std=2020 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include <sycl.hpp>

using namespace cl::sycl;

struct A {
  int a = 0;
  A() = default;
};
constexpr A THE_NAME;

template <auto &R> void temp() {}
template <auto &R> void foo(const char *out) {
  out = __builtin_unique_stable_name(temp<R>);
}

int main() {
  kernel_single_task<class kernel>(
      []() {
        const char *c;
        foo<THE_NAME>(c);
      });
}

// Note: the mangling here is actually the 'typeinfo name for void ()'.  That is
// because the type of temp<R> is actually the function type (which is void()).
// CHECK: @__builtin_unique_stable_name._Z3fooIL_ZL8THE_NAMEEEvPKc = private unnamed_addr addrspace(1) constant [9 x i8] c"_ZTSFvvE\00", align 1
