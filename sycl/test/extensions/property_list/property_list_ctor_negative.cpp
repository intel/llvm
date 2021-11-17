// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=1 2> %t_1.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-1 < %t_1.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=2 2> %t_2.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-2 < %t_2.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=3 2> %t_3.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-3 < %t_3.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=4 2> %t_4.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-4 < %t_4.err %s

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
#if (TEST_CASE == 1)
  auto InvalidPropertyList = sycl::ext::oneapi::property_list(1);
  // CHECK-ERROR-1: Unrecognized property in property list.
#elif (TEST_CASE == 2)
  auto InvalidPropertyList =
      sycl::ext::oneapi::property_list(sycl::property::no_init{}, true);
  // CHECK-ERROR-2: Unrecognized property in property list.
#elif (TEST_CASE == 3)
  auto InvalidPropertyList = sycl::ext::oneapi::property_list(
      sycl::property::no_init{}, sycl::property::no_init{});
  // CHECK-ERROR-3: Duplicate properties in property list.
#elif (TEST_CASE == 4)
  auto InvalidPropertyList = sycl::ext::oneapi::property_list(
      sycl::ext::oneapi::bar_v, sycl::ext::oneapi::bar_v);
  // CHECK-ERROR-4: Duplicate properties in property list.
#endif
}
