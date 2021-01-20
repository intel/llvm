// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

#include "sycl.hpp"

struct NestedStruct1 {
  struct NestedStruct2 {
    struct NestedStruct3 {};
  };
};

namespace {
struct StructInAnonymousNS {};
} // namespace

namespace ValidNS {
struct StructinValidNS {};
} // namespace ValidNS

struct Parent {
  using A = struct {
    struct Child1 {
      struct Child2 {};
    };
  };
};

struct MyWrapper {

public:
  void test() {
    cl::sycl::queue q;
    struct StructInsideFunc {};

    // no error
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<NestedStruct1::NestedStruct2::NestedStruct3>([] {});
    });

    // no error
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidNS::StructinValidNS>([] {});
    });

    // no error
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<Parent::A::Child1::Child2>([] {});
    });

    // expected-error@Inputs/sycl.hpp:220 {{'(anonymous namespace)::StructInAnonymousNS' is an invalid kernel name type}}
    // expected-note@Inputs/sycl.hpp:220 {{'(anonymous namespace)::StructInAnonymousNS' should be globally-visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<StructInAnonymousNS>([] {});
    });

    // expected-error@Inputs/sycl.hpp:220 {{'StructInsideFunc' is an invalid kernel name type}}
    // expected-note@Inputs/sycl.hpp:220 {{'StructInsideFunc' should be globally-visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<StructInsideFunc>([] {});
    });
  }
};

int main() {
  cl::sycl::queue q;

  return 0;
}
