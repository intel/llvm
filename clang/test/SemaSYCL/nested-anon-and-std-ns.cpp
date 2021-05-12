// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

// This test verifies that kernel names nested in 'std' and anonymous namespaces are correctly diagnosed.

#include "sycl.hpp"

namespace std {
namespace NestedInStd {
struct NestedStruct {};
}; // namespace NestedInStd
}; // namespace std

namespace ValidNS {
struct StructinValidNS {};
} // namespace ValidNS

struct ParentStruct {
  struct ChildStruct {
    int i;
  };
};

struct MyWrapper {

public:
  void test() {
    cl::sycl::queue q;

    // expected-error@#KernelSingleTask {{'std::NestedInStd::NestedStruct' is an invalid kernel name, 'std::NestedInStd::NestedStruct' is declared in the 'std' namespace}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::NestedInStd::NestedStruct>([] {});
    });

    // no error for valid ns
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidNS::StructinValidNS>([] {});
    });

    // expected-error@#KernelSingleTask {{'ParentStruct::ChildStruct' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ParentStruct::ChildStruct>([] {});
    });
  }
};
