// Ignore unexpected warnings because for some reason the warnings are emitted
// twice, e.g. once for `single_task`, then for `single_task<TestKernel0,
// (lambda at ...),
// sycl::ext::oneapi::experimental::properties<detail::properties_type_list<>>>`.
// RUN: %clangxx -fsycl -sycl-std=2020 -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning -Xclang -verify-ignore-unexpected=note %s -fsyntax-only -Wall -Wextra
#include <sycl/sycl.hpp>

using namespace sycl;
int main() {
  queue Q;
  event Ev;
  range<1> R1{1};
  range<2> R2(1, 1);
  range<3> R3(1, 1, 1);
  nd_range<1> NDR1{R1, R1};
  constexpr auto Props = sycl::ext::oneapi::experimental::properties{};

  // expected-warning@+1{{'single_task' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.single_task<class TestKernel0>(Props, []() {});
  // expected-warning@+1{{'single_task' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.single_task<class TestKernel1>(Ev, Props, []() {});
  // expected-warning@+1{{'single_task' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.single_task<class TestKernel2>({Ev}, Props, []() {});

  // expected-warning@+1{{'parallel_for' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.parallel_for<class TestKernel3>(NDR1, Props, [](nd_item<1>) {});

  // expected-warning@+2{{'single_task' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.single_task<class TestKernel4>(Props, []() {});
  });

  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel11>(R1, Props,
                                                    [](sycl::group<1>) {});
  });
  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel12>(R2, Props,
                                                    [](sycl::group<2>) {});
  });
  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel13>(R3, Props,
                                                    [](sycl::group<3>) {});
  });

  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel14>(R1, R1, Props,
                                                    [](sycl::group<1>) {});
  });
  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel15>(R2, R2, Props,
                                                    [](sycl::group<2>) {});
  });
  // expected-warning@+2{{'parallel_for_work_group' is deprecated: To specify properties, use a launch configuration object of type launch_config or a kernel functor with a get(sycl::ext::oneapi::experimental::properties_tag) member function instead.}}
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class TestKernel16>(R3, R3, Props,
                                                    [](sycl::group<3>) {});
  });
  return 0;
}
