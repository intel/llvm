// FIXME: Move to sycl/test.
// RUN: %{build} -fsyntax-only -o %t.out

#include <sycl/ext/oneapi/experimental/ballot_group.hpp>
#include <sycl/ext/oneapi/experimental/fixed_size_group.hpp>
#include <sycl/ext/oneapi/experimental/opportunistic_group.hpp>
#include <sycl/ext/oneapi/experimental/tangle_group.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

static_assert(
    syclex::is_user_constructed_group_v<syclex::ballot_group<sycl::sub_group>>);
static_assert(syclex::is_user_constructed_group_v<
              syclex::fixed_size_group<1, sycl::sub_group>>);
static_assert(syclex::is_user_constructed_group_v<
              syclex::fixed_size_group<2, sycl::sub_group>>);
static_assert(
    syclex::is_user_constructed_group_v<syclex::tangle_group<sycl::sub_group>>);
static_assert(syclex::is_user_constructed_group_v<syclex::opportunistic_group>);
