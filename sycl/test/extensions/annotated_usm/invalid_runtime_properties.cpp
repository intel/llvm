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
  TEST(malloc_device_annotated, N, q, properties{conduit, cache_config{large_slm}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_device_annotated, N, dev, Ctx, properties{conduit, foo{foo_enum::b}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_device_annotated<int>), N, q, properties{conduit, foz{0, 1}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_device_annotated<int>), N, dev, Ctx, properties{conduit, rt_prop1{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, q, properties{conduit, rt_prop2{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, dev, Ctx, properties{conduit, rt_prop3{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_device_annotated<int>), 1, N, q, properties{conduit, rt_prop4{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_device_annotated<int>), 1, N, dev, Ctx, properties{conduit, rt_prop5{}})



  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_host_annotated, N, q, properties{conduit, rt_prop6{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_host_annotated, N, Ctx, properties{conduit, rt_prop7{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_host_annotated<int>), N, q, properties{conduit, rt_prop8{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_host_annotated<int>), N, Ctx, properties{conduit, rt_prop9{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, q, properties{conduit, rt_prop10{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, Ctx, properties{conduit, rt_prop11{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_host_annotated<int>), 1, N, q, properties{conduit, rt_prop12{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_host_annotated<int>), 1, N, Ctx, properties{conduit, rt_prop13{}})



  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_shared_annotated, N, q, properties{conduit, rt_prop14{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_shared_annotated, N, dev, Ctx, properties{conduit, rt_prop15{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_shared_annotated<int>), N, q, properties{conduit, rt_prop16{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_shared_annotated<int>), N, dev, Ctx, properties{conduit, rt_prop17{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_shared_annotated, 1, N, q, properties{conduit, rt_prop18{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_shared_annotated, 1, N, dev, Ctx, properties{conduit, rt_prop19{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_shared_annotated<int>), 1, N, q, properties{conduit, rt_prop20{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_shared_annotated<int>), 1, N, dev, Ctx, properties{conduit, rt_prop21{}})
  


  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_annotated, N, q, alloc::device, properties{conduit, rt_prop22{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(malloc_annotated, N, q, alloc::device, properties{conduit, rt_prop23{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_annotated<int>), N, q, alloc::device, properties{conduit, rt_prop24{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((malloc_annotated<int>), N, q, alloc::device, properties{conduit, rt_prop25{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, q, alloc::device, properties{conduit, rt_prop26{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, q, alloc::device, properties{conduit, rt_prop27{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_annotated<int>), 1, N, q, alloc::device, properties{conduit, rt_prop28{}})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Found invalid runtime property in the property list.}}
  TEST((aligned_alloc_annotated<int>), 1, N, q, alloc::device, properties{conduit, rt_prop29{}})


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
