// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Expected failed tests for annotated USM allocation when given properties
// contain invalid compile-time property

#include <sycl/sycl.hpp>

#include "fake_properties.hpp"

#define TEST(f, args...)                                                       \
  { auto ap = f(args); }

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

// clang-format off

// Expect error when the property list contain invalid comile-time property
// Note that two errors are raised for each case, during:
// 1. validating malloc input properties
// 2. validating annotated_ptr properties
void testInvalidCompileTimeProperty(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  // expected-error@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* 24+ {{static assertion failed due to requirement 'is_valid_property_for_given_type': Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}bar_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated, N, q, properties{unaliased, bar})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated, N, dev, Ctx, properties{unaliased, alignment<4>, baz<5>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated<int>, N, q, properties{unaliased, baz<2>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}device_has_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated<int>, N, dev, Ctx, properties{alignment<4>, device_has<sycl::aspect::cpu>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}work_group_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, q, properties{unaliased, work_group_size<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, dev, Ctx, properties{unaliased, baz<3>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated<int>, 1, N, q, properties{unaliased, baz<0>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated<int>, 1, N, dev, Ctx, properties{unaliased, boo<void>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated, N, q, properties{unaliased, boo<int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated, N, Ctx, properties{alignment<4>, baz<2>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated<int>, N, q, properties{baz<7>, unaliased})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated<int>, N, Ctx, properties{unaliased, alignment<4>, boo<int, int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, q, properties{unaliased, boo<int, bool>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, Ctx, properties{unaliased, boo<int, int, int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated<int>, 1, N, q, properties{unaliased, alignment<4>, boo<bool>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated<int>, 1, N, Ctx, properties{unaliased, alignment<4>, boo<float>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}device_image_scope_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated, N, q, properties{unaliased, device_image_scope})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}host_access_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated, N, dev, Ctx, properties{alignment<4>, host_access_read})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated<int>, N, q, properties{boo<char>, unaliased})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_shared_annotated<int>, 1, N, dev, Ctx, properties{unaliased, alignment<4>, boo<int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}sub_group_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated, N, q, alloc::device, properties{unaliased, sub_group_size<2>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, q, alloc::device, properties{unaliased, boo<int, int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, dev, Ctx, alloc::device, properties{unaliased, boo<bool>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}work_group_size_hint_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated<int>, 1, N, q, alloc::device, properties{unaliased, work_group_size_hint<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated<int>, 1, N, dev, Ctx, alloc::device, properties{unaliased, alignment<4>, baz<5>})
}

// clang-format on
int main() {
  sycl::queue q;

  testInvalidCompileTimeProperty(q);
  return 0;
}
