// RUN: %{build} %cxx_std_optionc++23 -o %t.out
// RUN: %{run} %t.out

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/group_interface.hpp>

#include <sycl/builtins.hpp>

#include <type_traits>

namespace {

using namespace sycl;

static_assert(std::is_same_v<khr::work_group<1>::id_type, id<1>>);
static_assert(std::is_same_v<khr::work_group<1>::linear_id_type, size_t>);
static_assert(std::is_same_v<khr::work_group<1>::range_type, range<1>>);
#if defined(__cpp_lib_mdspan)
static_assert(
    std::is_same_v<khr::work_group<1>::extents_type, std::dextents<size_t, 1>>);
#endif
static_assert(std::is_same_v<khr::work_group<1>::size_type, size_t>);
static_assert(khr::work_group<1>::dimensions == 1);
static_assert(khr::work_group<1>::fence_scope == memory_scope::work_group);

static_assert(std::is_same_v<khr::work_group<2>::id_type, id<2>>);
static_assert(std::is_same_v<khr::work_group<2>::linear_id_type, size_t>);
static_assert(std::is_same_v<khr::work_group<2>::range_type, range<2>>);
#if defined(__cpp_lib_mdspan)
static_assert(
    std::is_same_v<khr::work_group<2>::extents_type, std::dextents<size_t, 2>>);
#endif
static_assert(std::is_same_v<khr::work_group<2>::size_type, size_t>);
static_assert(khr::work_group<2>::dimensions == 2);
static_assert(khr::work_group<2>::fence_scope == memory_scope::work_group);

static_assert(std::is_same_v<khr::work_group<3>::id_type, id<3>>);
static_assert(std::is_same_v<khr::work_group<3>::linear_id_type, size_t>);
static_assert(std::is_same_v<khr::work_group<3>::range_type, range<3>>);
#if defined(__cpp_lib_mdspan)
static_assert(
    std::is_same_v<khr::work_group<3>::extents_type, std::dextents<size_t, 3>>);
#endif
static_assert(std::is_same_v<khr::work_group<3>::size_type, size_t>);
static_assert(khr::work_group<3>::dimensions == 3);
static_assert(khr::work_group<3>::fence_scope == memory_scope::work_group);

void test_work_group_1d() {
  queue q;
  sycl::buffer<bool, 1> result(1);
  {
    sycl::host_accessor<bool, 0> acc{result};
    acc = true;
  }

  const int sz = 2;
  q.submit([&](handler &h) {
    sycl::accessor<bool, 0> acc{result, h};
    h.parallel_for(nd_range<1>{sz, sz}, [=](nd_item<1> item) {
      group<1> g = item.get_group();

      khr::work_group<1> wg = g;
      acc &= (wg.id() == g.get_group_id());
      acc &= (wg.linear_id() == g.get_group_linear_id());
      acc &= (wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
      acc &= (wg.extents().rank() == 1);
      acc &= (wg.extent(0) == g.get_local_range()[0]);
#endif
      acc &= (wg.size() == g.get_local_linear_range());

      khr::member_item wi = get_member_item(wg);
      acc &= (wi.id() == g.get_local_id());
      acc &= (wi.linear_id() == g.get_local_linear_id());
      acc &= (wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
      acc &= (wi.extents().rank() == 1);
      acc &= (wi.extent(0) == 1);
#endif
      acc &= (wi.size() == 1);
    });
  });

  q.wait();
  sycl::host_accessor<bool, 0> acc{result};
  assert(static_cast<bool>(acc));
}

void test_work_group_2d() {
  queue q;
  sycl::buffer<bool, 1> result(1);
  {
    sycl::host_accessor<bool, 0> acc{result};
    acc = true;
  }

  const int sz = 2;
  q.submit([&](handler &h) {
    sycl::accessor<bool, 0> acc{result, h};
    h.parallel_for(nd_range<2>{range<2>{sz, sz}, range<2>{sz, sz}},
                   [=](nd_item<2> item) {
                     group<2> g = item.get_group();

                     khr::work_group<2> wg = g;
                     acc &= (wg.id() == g.get_group_id());
                     acc &= (wg.linear_id() == g.get_group_linear_id());
                     acc &= (wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     acc &= (wg.extents().rank() == 2);
                     acc &= (wg.extent(0) == g.get_local_range()[0]);
                     acc &= (wg.extent(1) == g.get_local_range()[1]);
#endif
                     acc &= (wg.size() == g.get_local_linear_range());

                     khr::member_item wi = get_member_item(wg);
                     acc &= (wi.id() == g.get_local_id());
                     acc &= (wi.linear_id() == g.get_local_linear_id());
                     acc &= (wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
                     acc &= (wi.extents().rank() == 2);
                     acc &= (wi.extent(0) == 1);
                     acc &= (wi.extent(1) == 1);
#endif
                     acc &= (wi.size() == 1);
                   });
  });

  q.wait();
  sycl::host_accessor<bool, 0> acc{result};
  assert(static_cast<bool>(acc));
}

void test_work_group_3d() {
  queue q;
  sycl::buffer<bool, 1> result(1);
  {
    sycl::host_accessor<bool, 0> acc{result};
    acc = true;
  }

  const int sz = 2;
  q.submit([&](handler &h) {
    sycl::accessor<bool, 0> acc{result, h};
    h.parallel_for(nd_range<3>{range<3>{sz, sz, sz}, range<3>{sz, sz, sz}},
                   [=](nd_item<3> item) {
                     group<3> g = item.get_group();

                     khr::work_group<3> wg = g;
                     acc &= (wg.id() == g.get_group_id());
                     acc &= (wg.linear_id() == g.get_group_linear_id());
                     acc &= (wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     acc &= (wg.extents().rank() == 3);
                     acc &= (wg.extent(0) == g.get_local_range()[0]);
                     acc &= (wg.extent(1) == g.get_local_range()[1]);
                     acc &= (wg.extent(2) == g.get_local_range()[2]);
#endif
                     acc &= (wg.size() == g.get_local_linear_range());

                     khr::member_item wi = get_member_item(wg);
                     acc &= (wi.id() == g.get_local_id());
                     acc &= (wi.linear_id() == g.get_local_linear_id());
                     acc &= (wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
                     acc &= (wi.extents().rank() == 3);
                     acc &= (wi.extent(0) == 1);
                     acc &= (wi.extent(1) == 1);
                     acc &= (wi.extent(2) == 1);
#endif
                     acc &= (wi.size() == 1);
                   });
  });

  q.wait();
  sycl::host_accessor<bool, 0> acc{result};
  assert(static_cast<bool>(acc));
}

} // namespace

int main() {
  test_work_group_1d();
  test_work_group_2d();
  test_work_group_3d();
}
