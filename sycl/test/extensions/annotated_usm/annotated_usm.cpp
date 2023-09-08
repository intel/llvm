// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Compile-time tests for annotated USM allocation functions

#include "sycl/sycl.hpp"
#include <iostream>

// clang-format on

#define CHECK_TYPE(ap, T, PL)                                                  \
  static_assert(std::is_same_v<decltype(ap), annotated_ptr<T, decltype(PL)>>)

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

int main() {
  sycl::queue q;
  const sycl::context &Ctx = q.get_context();
  auto Dev = q.get_device();

  // Check alloc functions where usm_kind property is required in the input
  // property list All compile-time properties in the input properties appear on
  // the returned annotated_ptr, and runtime properties do not appear on the
  // returned annotated_ptr (i.e. `cache_config`)
  properties InP1{conduit, usm_kind<alloc::device>, cache_config{large_slm}};
  properties OutP1{conduit, usm_kind<alloc::device>};

  auto APtr1 = malloc_annotated<int>(N, q, InP1);
  CHECK_TYPE(APtr1 /*annotate_ptr*/, int /*allocated type*/,
             OutP1 /*returned property list*/);

  auto APtr11 = malloc_annotated<int, decltype(InP1)>(N, q, InP1);
  CHECK_TYPE(APtr11, int, OutP1);

  auto APtr12 =
      malloc_annotated<int, decltype(InP1), decltype(OutP1)>(N, Dev, Ctx, InP1);
  CHECK_TYPE(APtr12, int, OutP1);

  free(APtr1, q);
  free(APtr11, q);
  free(APtr12, q);

  // For alloc functions with `device` in the function name,
  // `usm_kind<alloc::device>` must appear on the returned annotated_ptr
  properties InP2{conduit, buffer_location<5>};
  properties OutP2{conduit, buffer_location<5>, usm_kind<alloc::device>};

  auto APtr2 = malloc_device_annotated<int>(N, q, InP2);
  CHECK_TYPE(APtr2, int, OutP2);

  auto APtr21 = aligned_alloc_device_annotated<int>(8, N, Dev, Ctx, InP2);
  CHECK_TYPE(APtr21, int, OutP2);

  free(APtr2, q);
  free(APtr21, q);

  // For alloc functions with `host` in the function name,
  // `usm_kind<alloc::host>` must appear on the returned annotated_ptr
  properties InP3{conduit, alignment<16>, usm_kind<alloc::host>};
  properties OutP3{conduit, alignment<16>, usm_kind<alloc::host>};

  auto APtr3 = aligned_alloc_host_annotated(1, N, Ctx, InP3);
  CHECK_TYPE(APtr3, void, OutP3);

  auto APtr31 = malloc_host_annotated<decltype(InP3)>(N, q);
  CHECK_TYPE(APtr31, void, OutP3);

  auto APtr32 =
      aligned_alloc_host_annotated<decltype(InP3), decltype(OutP3)>(0, N, q);
  CHECK_TYPE(APtr32, void, OutP3);

  free(APtr3, q);
  free(APtr31, q);
  free(APtr32, q);

  // For alloc functions with `shared` in the function name,
  // `usm_kind<alloc::shared>` must appear on the returned annotated_ptr
  properties OutP4{usm_kind<alloc::shared>};
  auto APtr4 = malloc_shared_annotated<int>(N, q);
  CHECK_TYPE(APtr4, int, OutP4);

  auto APtr41 = aligned_alloc_shared_annotated<int>(128, N, Dev, Ctx);
  CHECK_TYPE(APtr41, int, OutP4);

  auto APtr42 = malloc_shared_annotated(N, q);
  CHECK_TYPE(APtr42, void, OutP4);

  auto APtr43 = aligned_alloc_shared_annotated(16, N, q);
  CHECK_TYPE(APtr43, void, OutP4);

  free(APtr4, q);
  free(APtr41, q);
  free(APtr42, q);
  free(APtr43, q);

  // Check alloc functions with usm_kind argument and no usm_kind compile-time
  // property, usm_kind does not appear on the returned annotated_ptr
  auto APtr5 = malloc_annotated(N, Dev, Ctx, alloc::device);
  auto APtr51 = aligned_alloc_annotated<int>(1, N, q, alloc::device);
  static_assert(std::is_same_v<decltype(APtr5), annotated_ptr<void>>);
  static_assert(std::is_same_v<decltype(APtr51), annotated_ptr<int>>);
  free(APtr5, q);
  free(APtr51, q);

  return 0;
}
