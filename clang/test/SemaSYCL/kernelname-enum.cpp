// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s

#include "Inputs/sycl.hpp"

enum unscoped_enum_int : int {
  val_1,
  val_2
};

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
  void operator()() const {}
};

template <unscoped_enum_no_type_set EnumType>
class dummy_functor_2 {
public:
  void operator()() const {}
};

template <template <unscoped_enum_no_type_set EnumType> class C>
class templated_functor {
public:
  void operator()() const {}
};

template <scoped_enum_int EnumType>
class dummy_functor_3 {
public:
  void operator()() const {}
};

template <scoped_enum_no_type_set EnumType>
class dummy_functor_4 {
public:
  void operator()() const {}
};

int main() {

  dummy_functor_1<val_1> f1;
  dummy_functor_2<val_3> f2;
  dummy_functor_3<scoped_enum_int::val_2> f3;
  dummy_functor_4<scoped_enum_no_type_set::val_4> f4;
  templated_functor<dummy_functor_2> f5;

  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f1);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    // expected-error@#KernelSingleTask {{'dummy_functor_2<val_3>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{unscoped enum 'unscoped_enum_no_type_set' requires fixed underlying type}}
    // expected-note@+1{{in instantiation of function template specialization}}
    cgh.single_task(f2);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    // expected-error@#KernelSingleTask {{'templated_functor<dummy_functor_2>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{unscoped enum 'unscoped_enum_no_type_set' requires fixed underlying type}}
    // expected-note@+1{{in instantiation of function template specialization}}
    cgh.single_task(f5);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f3);
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task(f4);
  });

  return 0;
}
