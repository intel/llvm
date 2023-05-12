// The tests a basic E2E invoke_simd test checking that invoke_simd
// compiles and executes correctly on GPU, where the SIMD target is a
// ESIMD function.

// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip

// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

constexpr int VL = 16;

#ifndef INVOKE_SIMD
#define INVOKE_SIMD 1
#endif

constexpr bool use_invoke_simd = INVOKE_SIMD != 0;

__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, esimd::simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a + b;
}

// Use two functions with the same signature but different semantics
// called via invoke_simd for better testing.
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE1(float *A, simd<float, VL> b,
                                           int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE2(float *A, simd<float, VL> b,
                                           int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res + res;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_VOID(
    float *A, simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION;

float SPMD_CALLEE(float *A, float b, int i) { return A[i] + b; }

class ESIMDSelector : public device_selector {
  // Require GPU device
  virtual int operator()(const device &device) const {
    if (const char *dev_filter = getenv("ONEAPI_DEVICE_SELECTOR")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      std::cerr << "Supported 'ONEAPI_DEVICE_SELECTOR' env var values is "
                   "'*:gpu' and '"
                << filter_string << "' does not contain such substrings.\n";
      return -1;
    }
    // If "ONEAPI_DEVICE_SELECTOR" not defined, only allow gpu device
    return device.is_gpu() ? 1000 : -1;
  }
};

inline auto createExceptionHandler() {
  return [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << "sycl::exception: " << e0.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
      } catch (...) {
        std::cout << "generic exception\n";
      }
    }
  };
}

// Returns true if the test passed.
// If the template parameter use_func_directly is set to true, then this
// test verifies the case when function is passed directly to invoke_simd().
// Otherwise, the test verifices the case when invoke_simd accepts
// a reference to a variable holding address of the target function.
template <bool use_func_directly> bool test() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q(ESIMDSelector{}, createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running with use_func_directly = " << use_func_directly
            << " on " << dev.get_info<sycl::info::device::name>() << "\n";
  float *A = malloc_shared<float>(Size, q);
  float *B = malloc_shared<float>(Size, q);
  float *C = malloc_shared<float>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(
                                  VL)]] {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        float res = 0;

        if constexpr (use_invoke_simd) {
          if constexpr (use_func_directly) {
            // Pass SIMD callee directly to invoke_simd.
            res =
                invoke_simd(sg, SIMD_CALLEE1, uniform{A}, B[wi_id], uniform{i});
            res +=
                invoke_simd(sg, SIMD_CALLEE2, uniform{A}, B[wi_id], uniform{i});
            invoke_simd(sg, SIMD_CALLEE_VOID, uniform{A}, B[wi_id], uniform{i});
          } else {
            // Pass a reference to variable holding SIMD callee address.
            typedef simd<float, VL> __regcall (*FuncType)(float *,
                                                          simd<float, VL>, int);
            typedef void __regcall (*FuncVoidType)(float *, simd<float, VL>,
                                                   int);
            FuncType SIMD_CALLEE1_PTR = SIMD_CALLEE1;
            FuncType SIMD_CALLEE2_PTR = SIMD_CALLEE2;
            FuncVoidType SIMD_CALLEE_VOID_PTR = SIMD_CALLEE_VOID;
            res = invoke_simd(sg, SIMD_CALLEE1_PTR, uniform{A}, B[wi_id],
                              uniform{i});
            res += invoke_simd(sg, SIMD_CALLEE2_PTR, uniform{A}, B[wi_id],
                               uniform{i});
            invoke_simd(sg, SIMD_CALLEE_VOID_PTR, uniform{A}, B[wi_id],
                        uniform{i});
          }
        } else {
          res = SPMD_CALLEE(A, B[wi_id], wi_id);
          res += SPMD_CALLEE(A, B[wi_id], wi_id);
          res += SPMD_CALLEE(A, B[wi_id], wi_id);
        }
        C[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    return false;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (3 * (A[i] + B[i]) != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != 3*("
                  << A[i] << " + " << B[i] << ")\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt == 0;
}

int main() {
  bool Passed = true;
  Passed &= test<false>();
  Passed &= test<true>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}

SYCL_EXTERNAL
void __regcall SIMD_CALLEE_VOID(float *A, simd<float, VL> b,
                                int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
}
