// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-pc-windows-msvc -fsycl-is-device -verify -fsyntax-only -Wno-unused
// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-linux-gnu -fsycl-is-device -verify -fsyntax-only -Wno-unused
// Various Semantic analysis tests for the __builtin_unique_stable_id feature.

#include "Inputs/sycl.hpp"

struct S {}; // #SDecl
int f();

template<auto &S>
void wrapper() {
  __builtin_sycl_unique_stable_id(S);
}


static constexpr double global_double = 0;

template<typename T>
void usage_templ(T &t) {
  // expected-error@+2{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
  // expected-note@#usage_templ_instantiation{{in instantiation of function template specialization}}
  __builtin_sycl_unique_stable_id(t);
}

void usage(S s) {
  usage_templ(s); // #usage_templ_instantiation
  // expected-error@+2{{'S' does not refer to a value}}
  // expected-note@#SDecl{{declared here}}
  __builtin_sycl_unique_stable_id(S);
  // expected-error@+1{{expected variable name}}
  __builtin_sycl_unique_stable_id(f);

  // expected-error@+1{{expected variable name}}
  __builtin_sycl_unique_stable_id(f());

  // Needs to work.
  wrapper<global_double>();

  // expected-error@+1{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
  __builtin_sycl_unique_stable_id(s);
}

struct InAStruct {
  static const double static_member_double;
  const double member_double = 0;

  template<typename T>
  void templ_mem_func() {
    __builtin_sycl_unique_stable_id(static_member_double);
    // expected-error@+1{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
    __builtin_sycl_unique_stable_id(member_double);
  }

  template<typename T>
  void templ_mem_func2() {
    __builtin_sycl_unique_stable_id(T::static_member_double);

    T t{};
    // expected-error@+2{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
    // expected-note@#mem_func2_instantiation{{in instantiation of function template specialization}}
    __builtin_sycl_unique_stable_id(t.member_double);
  }

  void mem_func() {
    templ_mem_func<int>();
    templ_mem_func2<InAStruct>(); // #mem_func2_instantiation
    __builtin_sycl_unique_stable_id(static_member_double);
    // expected-error@+1{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
    __builtin_sycl_unique_stable_id(member_double);
  }
};

template<typename T>
struct DependentMembers {
  static const T static_member;
  const T member = 0;

  void test() {
    __builtin_sycl_unique_stable_id(static_member);
    // expected-error@+2{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
    // expected-note@#test_instantiation{{in instantiation of member function}}
    __builtin_sycl_unique_stable_id(member);
  }
};

void useDependentMembers() {
  DependentMembers<double> d;
  d.test(); // #test_instantiation

  __builtin_sycl_unique_stable_id(decltype(d)::static_member);
  // expected-error@+1{{argument passed to '__builtin_sycl_unique_stable_id' must have global storage}}
  __builtin_sycl_unique_stable_id(d.member);
}

// A few tests to ensure this gets correctly invalidated, like
// __builtin_sycl_unique_stable_name.
void invalidated() {
  (void)[]() {
    class K{};
    // Name gets changed because marking 'K' as a kernel changes the containing
    // lambda.
    static int GlobalStorageVar;
    constexpr const char *c = __builtin_sycl_unique_stable_id(GlobalStorageVar);
    // expected-error@+2{{kernel naming changes the result of an evaluated '__builtin_sycl_unique_stable_id'}}
    // expected-note@-2{{'__builtin_sycl_unique_stable_id' evaluated here}}
    __builtin_sycl_mark_kernel_name(K);
  };

  (void)[]() {
    // This name also gets changed, because naming 'lambda' causes the containg
    // lambda to have its name changed.
    static double ThisGlobalStorageVar;

    auto lambda = []() {};
    constexpr const char *d =
        __builtin_sycl_unique_stable_id(ThisGlobalStorageVar);

    // expected-error@#KernelSingleTaskFunc{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_id'}}
    // expected-note@#KernelSingleTask{{in instantiation of function template specialization}}
    // expected-note@+3{{in instantiation of function template specialization}}
    // expected-note@-5{{'__builtin_sycl_unique_stable_id' evaluated here}}
    cl::sycl::handler H;
    H.single_task(lambda);
  };
}
