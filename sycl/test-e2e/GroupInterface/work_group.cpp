// REQUIRES: cpu

// RUN: %{build} %cxx_std_optionc++23 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/khr/group_interface.hpp>

#include <sycl/builtins.hpp>

#include <type_traits>

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

int main() {
  queue q(cpu_selector_v);

  const int sz = 16;
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<1>{sz, sz}, [=](nd_item<1> item) {
      group<1> g = item.get_group();

      khr::work_group<1> wg = g;
      assert(wg.id() == g.get_group_id());
      assert(wg.linear_id() == g.get_group_linear_id());
      assert(wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
      assert(wg.extents().rank() == 1);
      assert(wg.extent(0) == g.get_local_range()[0]);
#endif
      assert(wg.size() == g.get_local_linear_range());

      khr::work_item wi = get_item(wg);
      assert(wi.id() == g.get_local_id());
      assert(wi.linear_id() == g.get_local_linear_id());
      assert(wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
      assert(wi.extents().rank() == 1);
      assert(wi.extent(0) == 1);
#endif
      assert(wi.size() == 1);
    });
  });
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<2>{range<2>{sz, sz}, range<2>{sz, sz}},
                   [=](nd_item<2> item) {
                     group<2> g = item.get_group();

                     khr::work_group<2> wg = g;
                     assert(wg.id() == g.get_group_id());
                     assert(wg.linear_id() == g.get_group_linear_id());
                     assert(wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     assert(wg.extents().rank() == 2);
                     assert(wg.extent(0) == g.get_local_range()[0]);
                     assert(wg.extent(1) == g.get_local_range()[1]);
#endif
                     assert(wg.size() == g.get_local_linear_range());

                     khr::work_item wi = get_item(wg);
                     assert(wi.id() == g.get_local_id());
                     assert(wi.linear_id() == g.get_local_linear_id());
                     assert(wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
                     assert(wi.extents().rank() == 2);
                     assert(wi.extent(0) == 1);
                     assert(wi.extent(1) == 1);
#endif
                     assert(wi.size() == 1);
                   });
  });
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<3>{range<3>{sz, sz, sz}, range<3>{sz, sz, sz}},
                   [=](nd_item<3> item) {
                     group<3> g = item.get_group();

                     khr::work_group<3> wg = g;
                     assert(wg.id() == g.get_group_id());
                     assert(wg.linear_id() == g.get_group_linear_id());
                     assert(wg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     assert(wg.extents().rank() == 3);
                     assert(wg.extent(0) == g.get_local_range()[0]);
                     assert(wg.extent(1) == g.get_local_range()[1]);
                     assert(wg.extent(2) == g.get_local_range()[2]);
#endif
                     assert(wg.size() == g.get_local_linear_range());

                     khr::work_item wi = get_item(wg);
                     assert(wi.id() == g.get_local_id());
                     assert(wi.linear_id() == g.get_local_linear_id());
                     assert(wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
                     assert(wi.extents().rank() == 3);
                     assert(wi.extent(0) == 1);
                     assert(wi.extent(1) == 1);
                     assert(wi.extent(2) == 1);
#endif
                     assert(wi.size() == 1);
                   });
  });
  q.wait();
}
