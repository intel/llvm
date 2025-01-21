// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"
// Test to ensue that we properly output the forward declarations of
// non-type-template-parameters when they refer to another template parameter.

// CHECK: Forward declarations of templated kernel function types:
// CHECK-NEXT: template <typename T, T nttp> struct SelfReferentialNTTP;
// CHECK-NEXT: template <typename U, int nttp> struct NonSelfRef;

template<typename T, T nttp>
struct SelfReferentialNTTP {};

using Foo = int;

template<typename U, Foo nttp>
struct NonSelfRef {};

void foo() {
  sycl::kernel_single_task<SelfReferentialNTTP<int, 1>>([](){});
  sycl::kernel_single_task<NonSelfRef<int, 1>>([](){});
}
