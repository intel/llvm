// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Annotated USM allocation tests that are expected to fail when:
// 1. required usm_kind is not provided in the property list
// 2. usm_kind in the property list conflicts with the function name

#include <sycl/sycl.hpp>

#include "fake_properties.hpp"

#define TEST(f, args...)                                                       \
  { auto ap = f(args); }

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

// clang-format off

// Missing usm kind in property list when it is required
void testMissingUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  properties InP{};
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST(malloc_annotated, N, q, properties{})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST(malloc_annotated, N, dev, Ctx, properties{alignment<8>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST((malloc_annotated<int>), N, q, properties{})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it as an argument or in the input property list.}}
  TEST((malloc_annotated<int>), N, dev, Ctx, properties{buffer_location<8>})
}

// Conflicting usm kinds between function name and property list
void testConflictingUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  TEST(malloc_shared_annotated, N, q, properties{usm_kind_host});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  TEST(malloc_shared_annotated, N, dev, Ctx, properties{usm_kind_device});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  TEST((malloc_shared_annotated<int>), N, q, properties{buffer_location<1>, usm_kind_host});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  TEST((malloc_shared_annotated<int>), N, dev, Ctx, properties{buffer_location<2>, usm_kind_device});
  
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_host_annotated'}}
  TEST(malloc_host_annotated, N, q, properties{usm_kind_shared, alignment<4>});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_host_annotated'}}
  TEST(malloc_host_annotated, N, Ctx, properties{usm_kind_device});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_host_annotated'}}
  TEST((malloc_host_annotated<int>), N, q, properties{buffer_location<1>, usm_kind_shared});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_host_annotated'}}
  TEST((malloc_host_annotated<int>), N, Ctx, properties{buffer_location<2>, usm_kind_device});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_device_annotated'}}
  TEST(malloc_device_annotated, N, q, properties{usm_kind_host});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_device_annotated'}}
  TEST(malloc_device_annotated, N, dev, Ctx, properties{usm_kind_shared});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_device_annotated'}}
  TEST((malloc_device_annotated<int>), N, q, properties{buffer_location<1>, usm_kind_host});

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_device_annotated'}}
  TEST((malloc_device_annotated<int>), N, dev, Ctx, properties{buffer_location<2>, usm_kind_shared});
}

// Duplicated properties (consistent or conflicting) exist in the property list
void testInvalidPropList() {
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Duplicate properties in property list.}}
  properties InvalidPropList{usm_kind_device, usm_kind_host};
}

// clang-format on

int main() {
  sycl::queue q;

  testMissingUsmKind(q);
  testConflictingUsmKind(q);
  testInvalidPropList();

  return 0;
}
