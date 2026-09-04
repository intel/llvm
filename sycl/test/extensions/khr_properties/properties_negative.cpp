// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=error,note %s
//
// Checks that misusing khr::properties produces the expected diagnostics.
// khr::properties holds its properties as private base classes, so misuse also
// produces base-class errors ("base specifier must name a class" / "base class
// specified more than once"); those are ignored here since we only assert that
// the friendly static_assert diagnostics are emitted.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/khr/properties.hpp>

namespace kd = sycl::khr::detail;
using namespace sycl::khr;

struct rt_key : kd::property_key_tag {};
struct rt : kd::property_base<rt_key> {
  int value;
  constexpr rt(int v = 0) : value(v) {}
};

void non_property() {
  // expected-error-re@sycl/khr/properties.hpp:* {{static assertion failed{{.*}}Template arguments of khr::properties must be properties.}}
  properties bad{5};
  (void)bad;
}

void duplicate_key() {
  // expected-error-re@sycl/khr/properties.hpp:* {{static assertion failed{{.*}}Duplicate properties in property list.}}
  properties bad{rt{1}, rt{2}};
  (void)bad;
}
