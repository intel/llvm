// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %BE_RUN_PLACEHOLDER %t.out
//
// Check that the host device is not included in devices returned by
// get_devices() if a non-host device type is specified.

#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

void check(info::device_type DT) {
  vector_class<device> Devices = device::get_devices(DT);
  for (const auto &Device : Devices)
    assert(!Device.is_host());
}

int main() {
  check(info::device_type::cpu);
  check(info::device_type::gpu);
  check(info::device_type::accelerator);
  check(info::device_type::custom);
  check(info::device_type::automatic);
}
