// REQUIRES: level_zero
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0 SYCL_DEVICE_FILTER=level_zero %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 SYCL_DEVICE_FILTER=level_zero %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=2 SYCL_DEVICE_FILTER=level_zero %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 SYCL_DEVICE_FILTER=level_zero %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that interleaving using copy and kernel operations are
// performed in-order, regardless of batching. IMPORTANT NOTE: this is a
// critical test, double-check if your changes are related to L0 events and
// links between commands.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>
#include <numeric>

static constexpr int MAGIC_NUM1 = 2;
static constexpr int buffer_size = 100;
sycl::usm::alloc AllocType = sycl::usm::alloc::device;

const size_t PartSize = 5;
const bool PartiallyPrint = buffer_size > 2 * PartSize;

void ValidationPrint(const std::string &vectName, const std::vector<int> &vect,
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

void RunCalculation(sycl::queue Q) {
  sycl::range<1> Range(buffer_size);
  auto Dev = Q.get_device();
  if (!Dev.has(sycl::aspect::usm_device_allocations))
    return;

  int *Dvalues =
      sycl::malloc<int>(buffer_size, Dev, Q.get_context(), AllocType);
  int *DvaluesTmp =
      sycl::malloc<int>(buffer_size, Dev, Q.get_context(), AllocType);

  std::vector<int> Hvalues1(buffer_size, 0);
  std::vector<int> HvaluesTmp(buffer_size, 0);
  std::iota(Hvalues1.begin(), Hvalues1.end(), 0);

  try {
    Q.memcpy(Dvalues, Hvalues1.data(), buffer_size * sizeof(int));

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (Dvalues[i] == i)
          Dvalues[i] += 1;
      });
    });

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (Dvalues[i] == i + 1)
          Dvalues[i] += 10;
      });
    });

    Q.memcpy(Hvalues1.data(), Dvalues, buffer_size * sizeof(int));
    Q.memcpy(DvaluesTmp, Hvalues1.data(), buffer_size * sizeof(int));

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (Dvalues[i] == i + 11)
          if (DvaluesTmp[i] == i + 11)
            Dvalues[i] += 100;
      });
    });

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (Dvalues[i] == i + 111)
          Dvalues[i] += 1000;
      });
    });

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Range, [=](sycl::item<1> itemID) {
        size_t i = itemID.get_id(0);
        if (Dvalues[i] == i + 1111)
          Dvalues[i] += 10000;
      });
    });

    Q.memcpy(Hvalues1.data(), Dvalues, buffer_size * sizeof(int));
    Q.memcpy(HvaluesTmp.data(), DvaluesTmp, buffer_size * sizeof(int));
    Q.wait();

    ValidationPrint("vector1[]  = ", Hvalues1,
                    [&](size_t i) { return i + 11111; });
    ValidationPrint("vector2[]  = ", HvaluesTmp,
                    [&](size_t i) { return i + 11; });

    for (int i = 0; i < buffer_size; ++i) {
      int expected = i + 11111;
      assert(Hvalues1[i] == expected);
    }

  } catch (sycl::exception &e) {
    std::cout << "Exception: " << std::string(e.what()) << std::endl;
  }

  free(Dvalues, Q);
  free(DvaluesTmp, Q);
}

int main(int argc, char *argv[]) {
  sycl::queue Q({sycl::property::queue::in_order{}});

  RunCalculation(Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
