// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER="" %t.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fsycl-targets=%sycl_triple -fpreview-breaking-changes %s -o %t.out %}
// ONEAPI_DEVICE_SELECTOR="*:-1" causes this test to not select any device at
// all.
// RUN: %if preview-breaking-changes-supported %{ env ONEAPI_DEVICE_SELECTOR="*:-1" %t.out %}

#include <sycl/sycl.hpp>
using namespace sycl;

int refuse_any_device_f(const device &d) { return -1; }

int main() {

  // Check exception message for custom device selector
  try {
    queue custom_queue(refuse_any_device_f);
  } catch (exception &E) {
    assert(std::string(E.what()).find("info::device_type::") ==
               std::string::npos &&
           "Incorrect device type in exception message for custom selector.");
  }

  // Check exception message for pre-defined devices
  try {
    queue gpu_queue(gpu_selector_v);
  } catch (exception &E) {
    assert(std::string(E.what()).find("info::device_type::gpu") !=
               std::string::npos &&
           "Incorrect device type in exception message for GPU device.");
  }
  try {
    queue cpu_queue(cpu_selector_v);
  } catch (exception &E) {
    assert(std::string(E.what()).find("info::device_type::cpu") !=
               std::string::npos &&
           "Incorrect device type in exception message for CPU device.");
  }
  try {
    queue acc_queue(accelerator_selector_v);
  } catch (exception &E) {
    assert(
        std::string(E.what()).find("info::device_type::accelerator") !=
            std::string::npos &&
        "Incorrect device type in exception message for Accelerator device.");
  }

  return 0;
}
