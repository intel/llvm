// REQUIRES: level_zero
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=2 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 ONEAPI_DEVICE_SELECTOR="level_zero:*" %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that the kernels are executed in-order, regardless of
// batching. IMPORTANT NOTE: this is a critical test, double-check if your
// changes are related to L0 events and links between commands.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <numeric>

static constexpr int MAGIC_NUM1 = 2;
static constexpr int buffer_size = 100;
sycl::usm::alloc AllocType = sycl::usm::alloc::shared;

const size_t PartSize = 5;
const bool PartiallyPrint = buffer_size > 2 * PartSize;

void ValidationPrint(const std::string &vectName, int *vect,
                     const std::function<int(size_t)> &ExpectedVal) {
  std::cerr << vectName;
  if (!PartiallyPrint) {
    for (size_t i = 0u; i < buffer_size; ++i) {
      std::cerr << " " << vect[i];
    }
  } else {
    for (size_t i = 0u; i < PartSize; ++i) {
      std::cerr << " " << vect[i];
    }
    std::cerr << " ... ";
    for (size_t i = buffer_size - PartSize; i < buffer_size; ++i) {
      std::cerr << " " << vect[i];
    }
  }

  std::cerr << std::endl << "expected[] = ";
  if (!PartiallyPrint) {
    for (size_t i = 0u; i < buffer_size; ++i) {
      std::cerr << " " << ExpectedVal(i);
    }
  } else {
    for (size_t i = 0u; i < PartSize; ++i) {
      std::cerr << " " << ExpectedVal(i);
    }
    std::cerr << " ... ";
    for (size_t i = buffer_size - PartSize; i < buffer_size; ++i) {
      std::cerr << " " << ExpectedVal(i);
    }
  }
  std::cerr << std::endl;
  for (int i = 0; i < buffer_size; ++i) {
    if (vect[i] != ExpectedVal(i)) {
      std::cerr << "i = " << i << " is wrong!!! " << std::endl;
      break;
    }
  }
  std::cerr << std::endl;
}

void IfTrueIncrementByValue(sycl::queue Q, sycl::range<1> Range, int *Harray,
                            int ValueToCheck, int ValueToIncrement) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class increment_usm>(Range, [=](sycl::item<1> itemID) {
      size_t i = itemID.get_id(0);
      if (Harray[i] == ValueToCheck) {
        Harray[i] += ValueToIncrement;
      }
    });
  });
}

void RunCalculation(sycl::queue Q) {
  sycl::range<1> Range(buffer_size);
  auto Dev = Q.get_device();
  if (!Dev.has(sycl::aspect::usm_shared_allocations))
    return;

  int *values = sycl::malloc<int>(buffer_size, Dev, Q.get_context(), AllocType);

  try {
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        values[i] = 1;
      });
    });

    IfTrueIncrementByValue(Q, Range, values, 1, 10);

    IfTrueIncrementByValue(Q, Range, values, 11, 100);

    IfTrueIncrementByValue(Q, Range, values, 111, 1000);

    IfTrueIncrementByValue(Q, Range, values, 1111, 10000);

    IfTrueIncrementByValue(Q, Range, values, 11111, 100000);

    Q.wait();

    ValidationPrint("vector[]  = ", values, [&](size_t i) { return 111111; });

    for (int i = 0; i < buffer_size; ++i) {
      int expected = 111111;
      assert(values[i] == expected);
    }

  } catch (sycl::exception &e) {
    std::cout << "Exception: " << std::string(e.what()) << std::endl;
  }

  free(values, Q);
}

int main(int argc, char *argv[]) {
  sycl::queue Q({sycl::property::queue::in_order{}});

  RunCalculation(Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
