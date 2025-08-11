// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/oneapi/experimental/chunk.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <sycl/ext/oneapi/experimental/tangle.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

template <typename Group>
inline constexpr bool is_user_constructed_group =
    syclex::is_user_constructed_group_v<Group>;

// is recognized as user-constructed
static_assert(is_user_constructed_group<syclex::fragment<sycl::sub_group>>);
static_assert(is_user_constructed_group<syclex::chunk<1, sycl::sub_group>>);
static_assert(is_user_constructed_group<syclex::chunk<2, sycl::sub_group>>);
static_assert(is_user_constructed_group<syclex::chunk<4, sycl::sub_group>>);
static_assert(is_user_constructed_group<syclex::chunk<8, sycl::sub_group>>);
static_assert(is_user_constructed_group<syclex::tangle<sycl::sub_group>>);

// sub_group itself is NOT user-constructed
static_assert(not is_user_constructed_group<sycl::sub_group>);
