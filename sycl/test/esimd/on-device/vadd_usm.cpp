// TODO enable on WIndows
// REQUIRES: linux
// REQUIRES: gpu
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

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
  constexpr unsigned Size = 1024;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 8;

  queue q(ESIMDSelector{}, exception_handler);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  // We need that many workitems. Each processes VL elements of data.
  cl::sycl::range<1> GlobalRange{Size / VL};
  // Number of workitems in each workgroup.
  cl::sycl::range<1> LocalRange{GroupSize};

  cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
      using namespace sycl::intel::gpu;

      int i = ndi.get_global_id(0);
      simd<float, VL> va = block_load<float, VL>(A + i * VL);
      simd<float, VL> vb = block_load<float, VL>(B + i * VL);
      simd<float, VL> vc = va + vb;
      block_store<float, VL>(C + i * VL, vc);
    });
  });
  e.wait();
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
