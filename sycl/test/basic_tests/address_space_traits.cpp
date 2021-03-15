// RUN: %clangxx -fsycl -fsycl-device-only -D__ENABLE_USM_ADDR_SPACE__ -fsycl-targets=%sycl_triple %s -c

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;
int main() {

  queue myQueue;
  myQueue.submit([&](handler &cgh) {
    cgh.single_task<class dummy>([=]() {
      static_assert(
          detail::deduce_AS<__attribute__((opencl_global)) int>::value ==
              access::address_space::global_space,
          "Unexpected address space");
      static_assert(
          detail::deduce_AS<__attribute__((opencl_local)) int>::value ==
              access::address_space::local_space,
          "Unexpected address space");
      static_assert(
          detail::deduce_AS<__attribute__((opencl_private)) int>::value ==
              access::address_space::private_space,
          "Unexpected address space");
      static_assert(
          detail::deduce_AS<__attribute__((opencl_constant)) int>::value ==
              access::address_space::constant_space,
          "Unexpected address space");
      static_assert(
          detail::deduce_AS<__attribute__((opencl_global_device)) int>::value ==
              access::address_space::global_device_space,
          "Unexpected address space");
      static_assert(
          detail::deduce_AS<__attribute__((opencl_global_host)) int>::value ==
              access::address_space::global_host_space,
          "Unexpected address space");
    });
  });
}
