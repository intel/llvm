// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Expected failed tests for annotated USM allocation:
// 1. Given properties contain invalid runtime property
// 2. Given properties contain invalid compile-time property
// 3. usm_kind in the property list conflicts with the function name
// 4. required usm_kind is not provided in the property list

#include <sycl/sycl.hpp>

#include "fake_properties.hpp"

#define TEST(f, args...)                                                       \
  { auto ap = f(args); }

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

// clang-format off

void testInvalidRuntimeProperty(sycl::queue &q) {
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_device_annotated<int>), N, q, properties{conduit, cache_config{large_slm}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_shared_annotated<int>), N, q, properties{conduit, foo{foo_enum::b}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_host_annotated<int>), N, q, properties{conduit, foz{0, 1}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_annotated<int>), N, q, alloc::device, properties{conduit, foo{foo_enum::a}, foz{0, 1}})
}

// Expect error when the property list contain invalid comile-time property
// Note that two errors are raised for each case, during:
// 1. validating malloc input properties
// 2. validating annotated_ptr properties
void testInvalidCompileTimeProperty(sycl::queue &q) {
  // expected-error-re@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated, N, q, properties{conduit, bar})

  // expected-error-re@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated, N, q, properties{conduit, baz<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated, N, q, properties{conduit, boo<double>})

  // expected-error-re@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated, N, q, alloc::device, properties{conduit, bar, baz<1>})
}

void testMissingUsmKind(sycl::queue &q) {
  // missing usm kind in property list when it is required
  properties InP{};
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST(malloc_annotated, N, q, InP)
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST((malloc_annotated<int>), N, q, InP)
}


void testConfilictingUsmKind(sycl::queue &q) {
  // Conflict usm kinds between function name and property list
  properties InP{usm_kind_host};
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  auto AP = malloc_shared_annotated<int, decltype(InP)>(N, q);
}

// clang-format on
int main() {
  sycl::queue q;

  testInvalidRuntimeProperty(q);
  testInvalidCompileTimeProperty(q);
  testMissingUsmKind(q);
  testConfilictingUsmKind(q);
  return 0;
}
