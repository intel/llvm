// REQUIRES: cpu

// RUN: %{build} %cxx_std_optionc++23 -o %t.out
// RUN: %{run} %t.out

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/group_interface.hpp>

#include <sycl/builtins.hpp>

#include <type_traits>

using namespace sycl;

static_assert(std::is_same_v<khr::sub_group::id_type, id<1>>);
static_assert(std::is_same_v<khr::sub_group::linear_id_type, uint32_t>);
static_assert(std::is_same_v<khr::sub_group::range_type, range<1>>);
#if defined(__cpp_lib_mdspan)
static_assert(
    std::is_same_v<khr::sub_group::extents_type, std::dextents<uint32_t, 1>>);
#endif
static_assert(std::is_same_v<khr::sub_group::size_type, uint32_t>);
static_assert(khr::sub_group::dimensions == 1);
static_assert(khr::sub_group::fence_scope == memory_scope::sub_group);

int main() {
  queue q(cpu_selector_v);

  const int sz = 16;
  q.submit([&](handler &h) {
    h.parallel_for(nd_range<1>{sz, sz}, [=](nd_item<1> item) {
      sub_group g = item.get_sub_group();

      khr::sub_group sg = g;
      assert(sg.id() == g.get_group_id());
      assert(sg.linear_id() == g.get_group_linear_id());
      assert(sg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
      assert(sg.extents().rank() == 1);
      assert(sg.extent(0) == g.get_local_range()[0]);
#endif
      assert(sg.size() == g.get_local_linear_range());
      assert(sg.max_size() == g.get_max_local_range()[0]);

      khr::member_item wi = get_member_item(sg);
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
                     sub_group g = item.get_sub_group();

                     khr::sub_group sg = g;
                     assert(sg.id() == g.get_group_id());
                     assert(sg.linear_id() == g.get_group_linear_id());
                     assert(sg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     assert(sg.extents().rank() == 1);
                     assert(sg.extent(0) == g.get_local_range()[0]);
#endif
                     assert(sg.size() == g.get_local_linear_range());
                     assert(sg.max_size() == g.get_max_local_range()[0]);

                     khr::member_item wi = get_member_item(sg);
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
    h.parallel_for(nd_range<3>{range<3>{sz, sz, sz}, range<3>{sz, sz, sz}},
                   [=](nd_item<3> item) {
                     sub_group g = item.get_sub_group();

                     khr::sub_group sg = g;
                     assert(sg.id() == g.get_group_id());
                     assert(sg.linear_id() == g.get_group_linear_id());
                     assert(sg.range() == g.get_group_range());
#if defined(__cpp_lib_mdspan)
                     assert(sg.extents().rank() == 1);
                     assert(sg.extent(0) == g.get_local_range()[0]);
#endif
                     assert(sg.size() == g.get_local_linear_range());
                     assert(sg.max_size() == g.get_max_local_range()[0]);

                     khr::member_item wi = get_member_item(sg);
                     assert(wi.id() == g.get_local_id());
                     assert(wi.linear_id() == g.get_local_linear_id());
                     assert(wi.range() == g.get_local_range());
#if defined(__cpp_lib_mdspan)
                     assert(wi.extents().rank() == 1);
                     assert(wi.extent(0) == 1);
#endif
                   });
  });
  q.wait();
}
