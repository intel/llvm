// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// Check that the code compiles with -O0 and -g
// RUN: %clangxx -I %sycl_include %s -o %t.out -fsycl -O0
// RUN: %clangxx -I %sycl_include %s -o %t.out -fsycl -O0 -g

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <iostream>
#include <string>

using namespace cl::sycl;

class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in SYCL_DEVICE_FILTER env
  virtual int operator()(const device &device) const {
    if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      if (filter_string.find("host") != std::string::npos)
        return device.is_host() ? 1000 : -1;
      std::cerr
          << "Supported 'SYCL_DEVICE_FILTER' env var values are 'gpu' and "
             "'host', '"
          << filter_string << "' does not contain such substrings.\n";
      return -1;
    }
    // If "SYCL_DEVICE_FILTER" not defined, only allow gpu device
    return device.is_gpu() ? 1000 : -1;
  }
};

auto exception_handler = [](exception_list l) {
  for (auto ep : l) {
    try {
      std::rethrow_exception(ep);
    } catch (cl::sycl::exception &e0) {
      std::cout << "sycl::exception: " << e0.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "std::exception: " << e.what() << std::endl;
    } catch (...) {
      std::cout << "generic exception\n";
    }
  }
};

int main(void) {
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 2;

  int A[Size];
  int B[Size];
  int C[Size] = {};

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  {
    cl::sycl::buffer<int, 1> bufA(A, Size);
    cl::sycl::buffer<int, 1> bufB(B, Size);
    cl::sycl::buffer<int, 1> bufC(C, Size);

    // We need that many task groups
    cl::sycl::range<1> GroupRange{Size / VL};

    // We need that many tasks in each group
    cl::sycl::range<1> TaskRange{GroupSize};

    cl::sycl::nd_range<1> Range{GroupRange, TaskRange};

    queue q(ESIMDSelector{}, exception_handler);
    q.submit([&](cl::sycl::handler &cgh) {
      auto accA = bufA.get_access<cl::sycl::access::mode::read>(cgh);
      auto accB = bufB.get_access<cl::sycl::access::mode::read>(cgh);
      auto accC = bufC.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::experimental::esimd;
            auto pA = accA.get_pointer().get();
            auto pB = accB.get_pointer().get();
            auto pC = accC.get_pointer().get();

            int i = ndi.get_global_id(0);
            constexpr int ESIZE = sizeof(int);
            simd<uint32_t, VL> offsets(0, ESIZE);

            simd<int, VL> va = gather<int, VL>(pA + i * VL, offsets);
            simd<int, VL> vb = block_load<int, VL>(pB + i * VL);
            simd<int, VL> vc = va + vb;

            block_store<int, VL>(pC + i * VL, vc);
          });
    });
  }
  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                << " + " << B[i] << "\n";
      return 1;
    }
  }

  std::cout << "Passed\n";
  return 0;
}
