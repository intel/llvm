// REQUIRES: level_zero
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=2 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
//
// The test is to check the execution of different queue operations has in-order
// semantics regardless of batching.
//
// IMPORTANT NOTE: this is a critical test, double-check if your changes are
// related to L0 barriers that provide links between commands within the same
// command-list or if your changes are related to L0 events and links between
// command-lists. if you have problems with this test, first see if other tests
// related to discard_events pass. And please check if the test passes without
// the discard_events property, if it doesn't pass then it's most likely a
// general issue unrelated to discard_events.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <numeric>

static constexpr int MAGIC_NUM1 = 2;

sycl::aspect getUSMAspect(sycl::usm::alloc Alloc) {
  if (Alloc == sycl::usm::alloc::host)
    return sycl::aspect::usm_host_allocations;

  if (Alloc == sycl::usm::alloc::device)
    return sycl::aspect::usm_device_allocations;

  assert(Alloc == sycl::usm::alloc::shared && "Unknown USM allocation type");
  return sycl::aspect::usm_shared_allocations;
}

void RunCalculation(sycl::queue queue, sycl::usm::alloc AllocType) {
  int buffer_size = 100;
  sycl::range<1> range(buffer_size);
  auto Dev = queue.get_device();
  if (!Dev.has(getUSMAspect(AllocType)))
    return;

  int *values1 =
      sycl::malloc<int>(buffer_size, Dev, queue.get_context(), AllocType);
  int *values2 =
      sycl::malloc<int>(buffer_size, Dev, queue.get_context(), AllocType);
  int *values3 =
      sycl::malloc<int>(buffer_size, Dev, queue.get_context(), AllocType);

  std::vector<int> values(buffer_size, 0);
  std::iota(values.begin(), values.end(), 0);

  std::vector<int> vec1(buffer_size, 0);
  std::vector<int> vec2(buffer_size, 0);
  std::vector<int> vec3(buffer_size, 0);

  try {
    queue.memcpy(values1, values.data(), buffer_size * sizeof(int));
    queue.memcpy(values2, values1, buffer_size * sizeof(int));
    queue.memcpy(values3, values2, buffer_size * sizeof(int));
    queue.memset(values1, 0, buffer_size * sizeof(int));

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class kernel1>(range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (values1[i] == 0)
          if (values2[i] == i)
            if (values3[i] == i) {
              values1[i] += i;
              values2[i] = MAGIC_NUM1;
              values3[i] = i;
            }
      });
    });

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class kernel2>(range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (values1[i] == i)
          if (values2[i] == MAGIC_NUM1)
            if (values3[i] == i) {
              values1[i] += 10;
            }
      });
    });

    queue.memcpy(values.data(), values1, buffer_size * sizeof(int));
    queue.memcpy(values2, values.data(), buffer_size * sizeof(int));

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class kernel3>(range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (values1[i] == i + 10)
          if (values2[i] == i + 10)
            if (values3[i] == i) {
              values1[i] += 100;
              values2[i] = i;
            }
      });
    });

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class kernel4>(range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (values1[i] == i + 110)
          if (values2[i] == i)
            if (values3[i] == i) {
              values1[i] += 1000;
            }
      });
    });

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class kernel5>(range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (values1[i] == i + 1110)
          if (values2[i] == i)
            if (values3[i] == i) {
              values1[i] += 10000;
            }
      });
    });

    queue.memcpy(vec1.data(), values1, buffer_size * sizeof(int));
    queue.memcpy(vec2.data(), values2, buffer_size * sizeof(int));
    queue.memcpy(vec3.data(), values3, buffer_size * sizeof(int));

    queue.wait();

    for (int i = 0; i < buffer_size; ++i) {
      int expected = i + 11110;
      assert(vec1[i] == expected);
      expected = i;
      assert(vec2[i] == expected);
      assert(vec3[i] == expected);
    }

  } catch (sycl::exception &e) {
    std::cout << "Exception: " << std::string(e.what()) << std::endl;
  }

  free(values1, queue);
  free(values2, queue);
  free(values3, queue);
}

int main(int argc, char *argv[]) {
  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue queue(sycl::default_selector{}, Props);

  RunCalculation(queue, sycl::usm::alloc::host);
  RunCalculation(queue, sycl::usm::alloc::shared);
  RunCalculation(queue, sycl::usm::alloc::device);

  std::cout << "The test passed." << std::endl;
  return 0;
}
