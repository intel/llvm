// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=1 2> %t_1.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-1 < %t_1.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=2 2> %t_2.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-2 < %t_2.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=3 2> %t_3.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-3 < %t_3.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=4 2> %t_4.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-4 < %t_4.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=5 2> %t_5.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-5 < %t_5.err %s
// RUN: not %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.bin -DTEST_CASE=6 2> %t_6.err
// RUN: FileCheck --check-prefix=CHECK-ERROR-6 < %t_6.err %s

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  auto EmptyPropertyList = sycl::ext::oneapi::property_list();
#if(TEST_CASE == 1)
  decltype(EmptyPropertyList)::get_property<sycl::ext::oneapi::boo>();
  // CHECK-ERROR-1: Property list does not contain the requested property.
#elif(TEST_CASE == 2)
  EmptyPropertyList.get_property<sycl::property::buffer::use_mutex>();
  // CHECK-ERROR-2: Property list does not contain the requested property.
#elif(TEST_CASE == 3)
  EmptyPropertyList.get_property<sycl::property::no_init>();
  // CHECK-ERROR-3: Property list does not contain the requested property.
#endif

  sycl::queue Q;
  auto PopulatedPropertyList = sycl::ext::oneapi::property_list(
      sycl::property::image::context_bound{Q.get_context()},
      sycl::ext::oneapi::bar_v,
      sycl::property::context::cuda::use_primary_context{});
#if(TEST_CASE == 4)
  decltype(PopulatedPropertyList)::get_property<sycl::ext::oneapi::boo>();
  // CHECK-ERROR-4: Property list does not contain the requested property.
#elif(TEST_CASE == 5)
  PopulatedPropertyList.get_property<sycl::property::buffer::use_mutex>();
  // CHECK-ERROR-5: Property list does not contain the requested property.
#elif(TEST_CASE == 6)
  PopulatedPropertyList.get_property<sycl::property::no_init>();
  // CHECK-ERROR-6: Property list does not contain the requested property.
#endif
}
