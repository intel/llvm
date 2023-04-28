// RUN: %clangxx -fsycl -fsyntax-only -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>
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
