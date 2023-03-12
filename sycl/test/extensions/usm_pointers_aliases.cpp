// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify %s -o %t.out
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  // Check device_ptr types.
  static_assert(
      std::is_same_v<
          ext::intel::device_ptr<void>,
          multi_ptr<void, access::address_space::ext_intel_global_device_space,
                    access::decorated::legacy>>,
      "Unexpected type for device_ptr<void>");
  static_assert(
      std::is_same_v<
          ext::intel::device_ptr<void, access::decorated::yes>,
          multi_ptr<void, access::address_space::ext_intel_global_device_space,
                    access::decorated::yes>>,
      "Unexpected type for device_ptr<void, access::decorated::yes>");
  static_assert(
      std::is_same_v<
          ext::intel::device_ptr<void, access::decorated::no>,
          multi_ptr<void, access::address_space::ext_intel_global_device_space,
                    access::decorated::no>>,
      "Unexpected type for device_ptr<void, access::decorated::no>");
  static_assert(
      std::is_same_v<ext::intel::decorated_device_ptr<void>,
                     ext::intel::device_ptr<void, access::decorated::yes>>,
      "Unexpected type for decorated_device_ptr<void>");
  static_assert(
      std::is_same_v<ext::intel::raw_device_ptr<void>,
                     ext::intel::device_ptr<void, access::decorated::no>>,
      "Unexpected type for raw_device_ptr<void>");

  // Check host_ptr types.
  static_assert(
      std::is_same_v<
          ext::intel::host_ptr<void>,
          multi_ptr<void, access::address_space::ext_intel_global_host_space,
                    access::decorated::legacy>>,
      "Unexpected type for host_ptr<void>");
  static_assert(
      std::is_same_v<
          ext::intel::host_ptr<void, access::decorated::yes>,
          multi_ptr<void, access::address_space::ext_intel_global_host_space,
                    access::decorated::yes>>,
      "Unexpected type for host_ptr<void, access::decorated::yes>");
  static_assert(
      std::is_same_v<
          ext::intel::host_ptr<void, access::decorated::no>,
          multi_ptr<void, access::address_space::ext_intel_global_host_space,
                    access::decorated::no>>,
      "Unexpected type for host_ptr<void, access::decorated::no>");
  static_assert(
      std::is_same_v<ext::intel::decorated_host_ptr<void>,
                     ext::intel::host_ptr<void, access::decorated::yes>>,
      "Unexpected type for decorated_host_ptr<void>");
  static_assert(
      std::is_same_v<ext::intel::raw_host_ptr<void>,
                     ext::intel::host_ptr<void, access::decorated::no>>,
      "Unexpected type for raw_host_ptr<void>");

  return 0;
}
