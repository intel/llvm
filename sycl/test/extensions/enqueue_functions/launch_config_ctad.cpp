// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

template <typename RangeT, typename PropertiesT>
static void Check(const RangeT &Range, const PropertiesT &Properties) {
  static_assert(
      std::is_same_v<decltype(oneapiext::launch_config{Range, Properties}),
                     oneapiext::launch_config<RangeT, PropertiesT>>);
}

template <typename RangeT> static void CheckRangeOnly(const RangeT &Range) {
  static_assert(
      std::is_same_v<
          decltype(oneapiext::launch_config{Range}),
          oneapiext::launch_config<RangeT, oneapiext::empty_properties_t>>);
}

template <typename RangeT> static void CheckAllForRange(const RangeT &Range) {
  CheckRangeOnly(Range);
  Check(Range, oneapiext::properties{});
  Check(Range, oneapiext::properties{oneapiext::work_group_progress<
                   oneapiext::forward_progress_guarantee::parallel,
                   oneapiext::execution_scope::root_group>});
}

int main() {
  CheckAllForRange(sycl::range<1>{1});
  CheckAllForRange(sycl::range<2>{1, 1});
  CheckAllForRange(sycl::range<3>{1, 1, 1});
  CheckAllForRange(sycl::nd_range<1>{{1}, {1}});
  CheckAllForRange(sycl::nd_range<2>{{1, 1}, {1, 1}});
  CheckAllForRange(sycl::nd_range<3>{{1, 1, 1}, {1, 1, 1}});
  return 0;
}
