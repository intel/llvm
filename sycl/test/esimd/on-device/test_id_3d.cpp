// TODO enable on WIndows
// REQUIRES: linux
// REQUIRES: gpu
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

// This test checks that multi-dimensional sycl::item can be used in ESIMD
// kernels.

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

int main(void) {
  constexpr unsigned LocalSizeX = 32;
  constexpr unsigned VL = 8;
  sycl::range<3> GroupRange{2, 3, 4};
  sycl::range<3> LocalRange{2, 3, LocalSizeX / VL};
  sycl::range<3> GlobalRange = GroupRange * LocalRange;
  sycl::range<3> ScalarGlobalRange = GlobalRange * sycl::range<3>{1, 1, VL};
  size_t Y = GlobalRange[1];
  size_t X = GlobalRange[2];

  queue q(ESIMDSelector{}, exception_handler);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  int *C = static_cast<int *>(
      malloc_shared(ScalarGlobalRange.size() * sizeof(int), dev, ctxt));

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(
        GlobalRange, [=](item<3> it) SYCL_ESIMD_KERNEL {
          using namespace sycl::intel::gpu;
          auto id = it.get_id();
          // calculate linear ID:
          size_t lin_id = id[0] * Y * X + id[1] * X + id[2];
          simd<int, VL> inc(0, 1);
          int off = (int)(lin_id * VL);
          simd<int, VL> val = inc + off;
          block_store<int, VL>(C + off, val);
        });
  });
  e.wait();
  int err_cnt = 0;

  for (size_t i = 0; i < ScalarGlobalRange.size(); ++i) {
    if (C[i] != (int)i) {
      if (++err_cnt < 10) {
        std::cerr << "*** ERROR at " << i << ": " << C[i] << "!=" << i
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "FAILED. " << err_cnt << "/" << ScalarGlobalRange.size()
              << " errors\n";
    return 1;
  }
  std::cout << "passed\n";
  return 0;
}
