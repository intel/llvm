// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests the sycl_khr_properties reduction property (initialize_to_identity):
// its traits and that the reduction() factory functions accept it.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/reduction.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct OtherClass {};

static_assert(is_property_v<kp::initialize_to_identity>);
static_assert(is_property_key_v<kp::key::initialize_to_identity>);
static_assert(!is_property_key_compile_time_v<kp::key::initialize_to_identity>);
static_assert(
    is_property_key_for_v<kp::key::initialize_to_identity, kp::tag::reduction>);
static_assert(
    is_property_for_v<kp::initialize_to_identity, kp::tag::reduction>);
static_assert(is_property_list_for_v<properties<kp::initialize_to_identity>,
                                     kp::tag::reduction>);
static_assert(!is_property_for_v<kp::initialize_to_identity, OtherClass>);
static_assert(!is_property_for_v<kp::initialize_to_identity, sycl::queue>);

void factories(int *usm, sycl::buffer<int, 1> buf, sycl::span<int, 4> span,
               sycl::handler &cgh) {
  // USM pointer, with and without identity.
  auto r1 =
      sycl::reduction(usm, sycl::plus<int>(), kp::initialize_to_identity{true});
  auto r2 = sycl::reduction(usm, 0, sycl::plus<int>(),
                            properties{kp::initialize_to_identity{}});
  // Buffer, with and without identity.
  auto r3 = sycl::reduction(buf, cgh, sycl::plus<int>(),
                            kp::initialize_to_identity{});
  auto r4 = sycl::reduction(buf, cgh, 0, sycl::plus<int>(),
                            properties{kp::initialize_to_identity{false}});
  // Fixed-extent span, with and without identity.
  auto r5 = sycl::reduction(span, sycl::plus<int>(),
                            properties{kp::initialize_to_identity{}});
  auto r6 =
      sycl::reduction(span, 0, sycl::plus<int>(), kp::initialize_to_identity{});
  // Empty property list.
  auto r7 = sycl::reduction(usm, sycl::plus<int>(), properties{});

  // Old-style construction must still resolve unambiguously.
  auto o1 = sycl::reduction(usm, sycl::plus<int>());
  auto o2 = sycl::reduction(usm, sycl::plus<int>(), sycl::property_list{});
  (void)r1;
  (void)r2;
  (void)r3;
  (void)r4;
  (void)r5;
  (void)r6;
  (void)r7;
  (void)o1;
  (void)o2;
}

// The reduction tag must not shadow sycl::reduction.
void unqualified(int *usm) {
  using namespace sycl;
  auto r = reduction(usm, plus<int>(), kp::initialize_to_identity{});
  (void)r;
}
