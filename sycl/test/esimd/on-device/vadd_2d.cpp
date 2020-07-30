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
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 8;
  constexpr unsigned GroupSize = 2;

  int A[Size];
  int B[Size];
  int C[Size] = {};

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  {
    cl::sycl::image<2> imgA(A, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});
    cl::sycl::image<2> imgB(B, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});
    cl::sycl::image<2> imgC(C, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});

    // We need that many workitems
    cl::sycl::range<1> GlobalRange{(Size / VL)};

    // Number of workitems in a workgroup
    cl::sycl::range<1> LocalRange{GroupSize};

    queue q(ESIMDSelector{}, exception_handler);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](cl::sycl::handler &cgh) {
      auto accA = imgA.get_access<uint4, cl::sycl::access::mode::read>(cgh);
      auto accB = imgB.get_access<uint4, cl::sycl::access::mode::read>(cgh);
      auto accC = imgC.get_access<uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::intel::gpu;

            constexpr int ESIZE = sizeof(int);
            int x = i * ESIZE * VL;
            int y = 0;

            simd<int, VL> va;
            auto va_ref = va.format<int, 1, VL>();
            va_ref = media_block_load<int, 1, VL>(accA, x, y);

            simd<int, VL> vb;
            auto vb_ref = vb.format<int, 1, VL>();
            vb_ref = media_block_load<int, 1, VL>(accB, x, y);

            simd<int, VL> vc;
            auto vc_ref = vc.format<int, 1, VL>();
            vc_ref = va_ref + vb_ref;
            media_block_store<int, 1, VL>(accC, x, y, vc_ref);
          });
    });
    e.wait();
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
