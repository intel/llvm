// RUN: %clangxx -D__ENABLE_USM_ADDR_SPACE__ -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void test(int *p) {
  // expected-error-re@sycl/access/access.hpp:* {{{{.*}}Not supported yet!}}
  std::ignore = dynamic_address_cast<
      sycl::access::address_space::ext_intel_global_device_space>(p);
  // expected-error-re@sycl/access/access.hpp:* {{{{.*}}Not supported yet!}}
  std::ignore = dynamic_address_cast<
      sycl::access::address_space::ext_intel_global_host_space>(p);
}
