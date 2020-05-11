// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"

enum class no_namespace_int : int {
  val_1,
  val_2
};

enum class no_namespace_short : short {
  val_1,
  val_2
};

namespace internal {
enum class namespace_short : short {
  val_1,
  val_2
};
}

template <no_namespace_int EnumType>
class dummy_functor_1 {
public:
  void operator()() {}
};

template <no_namespace_short EnumType>
class dummy_functor_2 {
public:
  void operator()() {}
};

template <internal::namespace_short EnumType>
class dummy_functor_3 {
public:
  void operator()() {}
};

int main() {

  dummy_functor_1<no_namespace_int::val_1> f1;
  dummy_functor_2<no_namespace_short::val_2> f2;
  dummy_functor_3<internal::namespace_short::val_2> f3;

  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f1);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f2);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f3);
  });

  return 0;
}

// CHECK: Forward declarations of templated kernel function types:
// CHECK: enum class no_namespace_int : int;
// CHECK: template <no_namespace_int EnumType> class dummy_functor_1;
// CHECK: enum class no_namespace_short : short;
// CHECK: template <no_namespace_short EnumType> class dummy_functor_2;
// CHECK: namespace internal {
// CHECK-NEXT: enum class namespace_short : short;
// CHECK-NEXT: }
// CHECK: template <internal::namespace_short EnumType> class dummy_functor_3;

// CHECK: Specializations of KernelInfo for kernel function types:
// CHECK: template <> struct KernelInfo<dummy_functor_1<(no_namespace_int)0>>
// CHECK: template <> struct KernelInfo<dummy_functor_2<(no_namespace_short)1>>
// CHECK: template <> struct KernelInfo<dummy_functor_3<(internal::namespace_short)1>>
