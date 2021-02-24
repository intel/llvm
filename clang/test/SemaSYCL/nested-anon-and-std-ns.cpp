// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

#include "sycl.hpp"

namespace std {
namespace NestedInStd {
struct NestedStruct {};
}; // namespace NestedInStd
}; // namespace std

namespace NestedInStd {
namespace std {
struct NestedStruct {};
}; // namespace std
}; // namespace NestedInStd

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

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::NestedInStd::NestedStruct>([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<NestedInAnon::StructInAnonymousNS>([] {});
    });

    // no error for valid ns
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidNS::StructinValidNS>([] {});
    });
  }
};
