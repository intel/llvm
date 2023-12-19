// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Expected failed tests for annotated USM allocation when given properties
// contain invalid compile-time property

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/fpga_extensions.hpp>

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

  // expected-error@sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp:* 33+ {{static assertion failed due to requirement 'is_valid_property_for_given_type': Property is invalid for the given type.}}
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}bar_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated, N, q, properties{conduit, bar})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated, N, dev, Ctx, properties{conduit, buffer_location<1>, baz<5>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated<int>, N, q, properties{conduit, baz<2>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}device_has_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_device_annotated<int>, N, dev, Ctx, properties{buffer_location<1>, device_has<sycl::aspect::cpu>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}work_group_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, q, properties{register_map, work_group_size<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated, 1, N, dev, Ctx, properties{register_map, baz<3>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}baz_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated<int>, 1, N, q, properties{register_map, baz<0>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_device_annotated<int>, 1, N, dev, Ctx, properties{conduit, boo<void>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated, N, q, properties{conduit, boo<int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}streaming_interface_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated, N, Ctx, properties{buffer_location<1>, streaming_interface_accept_downstream_stall})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}register_map_interface_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated<int>, N, q, properties{register_map_interface_wait_for_done_write, register_map})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_host_annotated<int>, N, Ctx, properties{conduit, buffer_location<1>, boo<int, int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, q, properties{register_map, boo<int, bool>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated, 1, N, Ctx, properties{register_map, boo<int, int, int>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated<int>, 1, N, q, properties{register_map, buffer_location<1>, boo<bool>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_host_annotated<int>, 1, N, Ctx, properties{register_map, buffer_location<1>, boo<float>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}device_image_scope_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated, N, q, properties{conduit, device_image_scope})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}host_access_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated, N, dev, Ctx, properties{buffer_location<1>, host_access_read})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}boo_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated<int>, N, q, properties{boo<char>, register_map})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}init_mode_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_shared_annotated<int>, N, dev, Ctx, properties{conduit, buffer_location<1>, init_mode_reset})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}implement_in_csr_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_shared_annotated, 1, N, q, properties{register_map, implement_in_csr_on})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}ready_latency_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_shared_annotated, 1, N, dev, Ctx, properties{register_map, ready_latency<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}bits_per_symbol_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_shared_annotated<int>, 1, N, q, properties{register_map, bits_per_symbol<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}pipelined_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_shared_annotated<int>, 1, N, dev, Ctx, properties{register_map, buffer_location<1>, pipelined<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}sub_group_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated, N, q, alloc::device, properties{conduit, sub_group_size<2>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}uses_valid_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated, N, dev, Ctx, alloc::device, properties{buffer_location<1>, uses_valid_on})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}first_symbol_in_high_order_bits_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated<int>, N, q, alloc::device, properties{first_symbol_in_high_order_bits_on, register_map})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}protocol_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(malloc_annotated<int>, N, dev, Ctx, alloc::device, properties{conduit, buffer_location<1>, protocol_avalon_streaming_uses_ready})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}word_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, q, alloc::device, properties{register_map, word_size<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}stride_size_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated, 1, N, dev, Ctx, alloc::device, properties{register_map, stride_size<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}work_group_size_hint_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated<int>, 1, N, q, alloc::device, properties{register_map, work_group_size_hint<1>})

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}latency_anchor_id_key{{.+}}: Found invalid compile-time property in the property list.}}
  TEST(aligned_alloc_annotated<int>, 1, N, dev, Ctx, alloc::device, properties{register_map, buffer_location<1>, latency_anchor_id<5>})
}

// clang-format on
int main() {
  sycl::queue q;

  testInvalidCompileTimeProperty(q);
  return 0;
}
