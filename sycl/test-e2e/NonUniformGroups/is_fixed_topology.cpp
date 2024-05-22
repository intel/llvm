// RUN: %{build} -fsyntax-only -o %t.out

#include <sycl/detail/core.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

#ifdef SYCL_EXT_ONEAPI_ROOT_GROUP
static_assert(syclex::is_fixed_topology_group_v<syclex::root_group<1>>);
static_assert(syclex::is_fixed_topology_group_v<syclex::root_group<2>>);
static_assert(syclex::is_fixed_topology_group_v<syclex::root_group<3>>);
#endif
static_assert(syclex::is_fixed_topology_group_v<sycl::group<1>>);
static_assert(syclex::is_fixed_topology_group_v<sycl::group<2>>);
static_assert(syclex::is_fixed_topology_group_v<sycl::group<3>>);
static_assert(syclex::is_fixed_topology_group_v<sycl::sub_group>);
