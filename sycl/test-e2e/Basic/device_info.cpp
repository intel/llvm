// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/17614

// Test checks ext_intel_fan_speed, ext_intel_power_limits and
// ext_intel_current_clock_throttle_reasons device descriptors.

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;
  device dev = q.get_device();
  if (dev.has(aspect::ext_intel_fan_speed)) {
    auto fan_speed = dev.get_info<ext::intel::info::device::fan_speed>();
    std::cout << "Fan speed: " << fan_speed << std::endl;
  } else {
    bool exception_thrown = false;
    try {
      std::ignore = dev.get_info<ext::intel::info::device::fan_speed>();
    } catch (const sycl::exception &e) {
      assert(e.code() == sycl::errc::feature_not_supported);
      exception_thrown = true;
    }
    assert(exception_thrown && "Exception not thrown");
  }

  if (dev.has(aspect::ext_intel_power_limits)) {
    auto min_limit = dev.get_info<ext::intel::info::device::min_power_limit>();
    auto max_limit = dev.get_info<ext::intel::info::device::max_power_limit>();
    std::cout << "Min power limit: " << min_limit << std::endl;
    std::cout << "Max power limit: " << max_limit << std::endl;
  } else {
    bool exception_thrown = false;
    try {
      std::ignore = dev.get_info<ext::intel::info::device::min_power_limit>();
      std::ignore = dev.get_info<ext::intel::info::device::max_power_limit>();
    } catch (const sycl::exception &e) {
      assert(e.code() == sycl::errc::feature_not_supported);
      exception_thrown = true;
    }
    assert(exception_thrown && "Exception not thrown");
  }

  if (dev.has(aspect::ext_intel_current_clock_throttle_reasons)) {
    auto throttle_reasons = dev.get_info<
        ext::intel::info::device::current_clock_throttle_reasons>();
    if (throttle_reasons.empty()) {
      std::cout << "No throttling" << std::endl;
    } else {
      std::cout << "Throttling reasons:" << std::endl;
      for (auto reason : throttle_reasons) {
        if (reason == ext::intel::throttle_reason::power_cap) {
          std::cout << "Power cap" << std::endl;
        } else if (reason == ext::intel::throttle_reason::current_limit) {
          std::cout << "Current limit" << std::endl;
        } else if (reason == ext::intel::throttle_reason::thermal_limit) {
          std::cout << "Thermal limit" << std::endl;
        } else if (reason == ext::intel::throttle_reason::psu_alert) {
          std::cout << "PSU alert" << std::endl;
        } else if (reason == ext::intel::throttle_reason::sw_range) {
          std::cout << "SW range" << std::endl;
        } else if (reason == ext::intel::throttle_reason::hw_range) {
          std::cout << "HW range" << std::endl;
        } else if (reason == ext::intel::throttle_reason::other) {
          std::cout << "Other" << std::endl;
        } else {
          assert(false && "Unknown throttle reason");
        }
      }
    }
  } else {
    bool exception_thrown = false;
    try {
      std::ignore = dev.get_info<
          ext::intel::info::device::current_clock_throttle_reasons>();
    } catch (const sycl::exception &e) {
      assert(e.code() == sycl::errc::feature_not_supported);
      exception_thrown = true;
    }
    assert(exception_thrown && "Exception not thrown");
  }

  return 0;
}
