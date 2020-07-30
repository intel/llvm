// TODO enable on WIndows
// REQUIRES: linux
// REQUIRES: gpu
// RUN: %clangxx-esimd -fsycl -D_CRT_SECURE_NO_WARNINGS=1 %s -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

// This test checks that accessor-based memory accesses work correctly in ESIMD.

#include <CL/sycl.hpp>
#include <CL/sycl/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;

class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in SYCL_DEVICE_TYPE env
  virtual int operator()(const device &device) const {
    if (const char *dev_type = getenv("SYCL_DEVICE_TYPE")) {
      if (!strcmp(dev_type, "GPU"))
        return device.is_gpu() ? 1000 : -1;
      if (!strcmp(dev_type, "HOST"))
        return device.is_host() ? 1000 : -1;
      std::cerr << "Supported 'SYCL_DEVICE_TYPE' env var values are 'GPU' and "
                   "'HOST', '"
                << dev_type << "' is not.\n";
      return -1;
    }
    // If "SYCL_DEVICE_TYPE" not defined, only allow gpu device
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

constexpr unsigned int VL = 1024 * 128;

using Ty = float;

int main() {
  Ty data0[VL] = {0};
  Ty data1[VL] = {0};
  constexpr Ty VAL = 5;

  for (int i = 0; i < VL; i++) {
    data0[i] = i;
  }

  try {
    queue q(ESIMDSelector{}, exception_handler);

    buffer<Ty, 1> buf0(data0, range<1>(VL));
    buffer<Ty, 1> buf1(data1, range<1>(VL));

    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";

      auto acc0 = buf0.get_access<access::mode::read_write>(cgh);
      auto acc1 = buf1.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          range<1>(1), [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::intel::gpu;
            unsigned int offset = 0;
            for (int k = 0; k < VL / 16; k++) {
              simd<Ty, 16> var = block_load<Ty, 16>(acc0, offset);
              var += VAL;
              block_store(acc0, offset, var);
              block_store(acc1, offset, var + 1);
              offset += 64;
            }
          });
    });

    q.wait();

  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  int err_cnt = 0;

  for (int i = 0; i < VL; i++) {
    Ty gold0 = i + VAL;
    Ty gold1 = gold0 + 1;
    Ty val0 = data0[i];
    Ty val1 = data1[i];

    if (val0 != gold0) {
      if (++err_cnt < 10)
        std::cerr << "*** ERROR at data0[" << i << "]: " << val0
                  << " != " << gold0 << "(gold)\n";
    }
    if (val1 != gold1) {
      if (++err_cnt < 10)
        std::cerr << "*** ERROR at data1[" << i << "]: " << val1
                  << " != " << gold1 << "(gold)\n";
    }
  }
  if (err_cnt == 0) {
    std::cout << "Passed\n";
    return 0;
  } else {
    std::cout << "Failed: " << err_cnt << " of " << VL << " errors\n";
    return 1;
  }
}
