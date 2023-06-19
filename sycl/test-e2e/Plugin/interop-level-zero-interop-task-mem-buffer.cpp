// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test for Level Zero interop_task for buffer.
// Level-Zero
#include <iostream>
#include <level_zero/ze_api.h>
// SYCL
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t SIZE = 16;

int main() {
  queue queue{};

  try {
    buffer<uint8_t, 1> buffer(SIZE);

    ze_context_handle_t ze_context =
        get_native<backend::ext_oneapi_level_zero>(queue.get_context());

    queue
        .submit([&](handler &cgh) {
          auto buffer_acc = buffer.get_access<access::mode::write>(cgh);
          cgh.interop_task([=](const interop_handler &ih) {
            void *device_ptr =
                ih.get_mem<backend::ext_oneapi_level_zero>(buffer_acc);
            ze_memory_allocation_properties_t memAllocProperties{};
            memAllocProperties.stype =
                ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
            ze_result_t res = zeMemGetAllocProperties(
                ze_context, device_ptr, &memAllocProperties, nullptr);
            assert(res == ZE_RESULT_SUCCESS);
          });
        })
        .wait();
  } catch (exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  } catch (const char *msg) {
    std::cout << "Exception caught: " << msg << std::endl;
    return 1;
  }

  return 0;
}
