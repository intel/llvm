// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Expected failed tests for annotated USM allocation when given properties
// contain invalid runtime property

#include <sycl/sycl.hpp>

#include "fake_properties.hpp"

#define TEST(f, args...)                                                       \
  { auto ap = f(args); }

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

// Expect error when the property list contain invalid runtime property
// Note a runtime property is valid for annotated USM allocation only when the
// type trait `detail::IsRuntimePropertyValid` is specialized with the property
void testInvalidRuntimeProperty(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  // clang-format off

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_annotated, N, q, properties{usm_kind_device, rt_prop30{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_annotated, N, dev, Ctx, properties{usm_kind_device, rt_prop31{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_annotated<int>), N, q, properties{usm_kind_device, rt_prop32{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_annotated<int>), N, dev, Ctx, properties{usm_kind_device, rt_prop33{}})

  // clang-format on
}

int main() {
  sycl::queue q;

  testInvalidRuntimeProperty(q);
  return 0;
}
