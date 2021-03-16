// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

// This test verifies that kernel names nested in 'std' and anonymous namespaces are correctly diagnosed.

#include "sycl.hpp"

namespace std {
namespace NestedInStd {
struct NestedStruct {};
}; // namespace NestedInStd
}; // namespace std

namespace {
namespace NestedInAnon {
struct StructInAnonymousNS {};
} // namespace NestedInAnon
} // namespace

namespace Named {
namespace {
struct IsThisValid {};
} // namespace
} // namespace Named

namespace ValidNS {
struct StructinValidNS {};
} // namespace ValidNS

struct MyWrapper {

public:
  void test() {
    cl::sycl::queue q;

    // expected-error@Inputs/sycl.hpp:220 {{'std::NestedInStd::NestedStruct' is an invalid kernel name, 'std::NestedInStd::NestedStruct' is declared in the 'std' namespace}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::NestedInStd::NestedStruct>([] {});
    });

    // expected-error@Inputs/sycl.hpp:220 {{'Named::(anonymous namespace)::IsThisValid' should be globally-visibl}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<Named::IsThisValid>([] {});
    });

    // expected-error@Inputs/sycl.hpp:220 {{'(anonymous namespace)::NestedInAnon::StructInAnonymousNS' should be globally-visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<NestedInAnon::StructInAnonymousNS>([] {});
    });

    // no error for valid ns
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidNS::StructinValidNS>([] {});
    });
  }
};
