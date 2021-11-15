// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::experimental::esimd;
constexpr int VL = 16;

__attribute__((always_inline))
esimd::simd<float, VL> ESIMD_CALLEE(float *A, esimd::simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a + b;
}

SYCL_EXTERNAL
simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res;
}

float SPMD_CALLEE(float *A, float b, int i) {
  return A[i] + b;
}

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

inline auto createExceptionHandler() {
  return [](exception_list l) {
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
}

#ifndef INVOKE_SIMD
#define INVOKE_SIMD 1
#endif

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4*VL;

  queue q(ESIMDSelector{}, createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  // TODO: release memory in the end of the test
  float *A =
    static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B =
    static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
    static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  cl::sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  cl::sycl::range<1> LocalRange{GroupSize};

  cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
      sub_group sg = ndi.get_sub_group();
      group<1> g = ndi.get_group();
      uint32_t i = sg.get_group_linear_id() * VL + g.get_linear_id() * GroupSize;
      uint32_t wi_id = i + sg.get_local_id();

#if INVOKE_SIMD != 0
        float res = invoke_simd(sg, SIMD_CALLEE, uniform{ A }, B[wi_id], uniform{ i });
#else
        float res = SPMD_CALLEE(A, B[wi_id], wi_id);
#endif
        C[wi_id] = res;
      });
    });
    e.wait();
  }
  catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.get_cl_code();
  }

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
