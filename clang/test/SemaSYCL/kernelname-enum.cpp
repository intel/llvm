// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s

#include "sycl.hpp"

enum unscoped_enum_int : int {
  val_1,
  val_2
};

// expected-note@+1 {{'unscoped_enum_no_type_set' declared here}}
enum unscoped_enum_no_type_set {
  val_3,
  val_4
};

enum class scoped_enum_int : int {
  val_1,
  val_2
};

enum class scoped_enum_no_type_set {
  val_3,
  val_4
};

template <unscoped_enum_int EnumType>
class dummy_functor_1 {
public:
  void operator()() {}
};

// expected-error@+2 {{kernel name is invalid. Unscoped enum requires fixed underlying type}}
template <unscoped_enum_no_type_set EnumType>
class dummy_functor_2 {
public:
  void operator()() {}
};

template <scoped_enum_int EnumType>
class dummy_functor_3 {
public:
  void operator()() {}
};

template <scoped_enum_no_type_set EnumType>
class dummy_functor_4 {
public:
  void operator()() {}
};

int main() {

  dummy_functor_1<val_1> f1;
  dummy_functor_2<val_3> f2;
  dummy_functor_3<scoped_enum_int::val_2> f3;
  dummy_functor_4<scoped_enum_no_type_set::val_4> f4;

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

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f4);
  });

  return 0;
}
