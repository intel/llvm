// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// RUN: not %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out 2>&1 | FileCheck %s
//
// The spec only allows scalar types
// (http://eel.is/c++draft/basic.types.general#def:type,scalar)
// TODO FIXME: Compile must fail with meaningful error message, but currently it
// compiles with no error
// XFAIL: gpu
/*
 * Test case specification: Test and report errors if accessor argument is
 * passed to invoked ESIMD function
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

template <typename AccessorTy>
ESIMD_INLINE esimd::simd<float, VL> ESIMD_CALLEE(AccessorTy acc,
                                                 esimd::simd<float, VL> b,
                                                 int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(acc, i * sizeof(float));
  return a + b;
}

template <typename AccessorTy>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(AccessorTy acc, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION;

using namespace sycl;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctx = q.get_context();

  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctx));
  float *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctx));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctx));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    buffer<float, 1> buf(A, range<1>(Size));

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class Test>(
          Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i = sg.get_group_linear_id() * VL +
                         g.get_group_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();

            float res = invoke_simd(sg, SIMD_CALLEE<decltype(acc)>,
                                    uniform{acc}, B[wi_id], uniform{i});
            // CHECK: TODO FIXME
            C[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, ctx);
    sycl::free(B, ctx);
    sycl::free(C, ctx);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
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

  sycl::free(A, ctx);
  sycl::free(B, ctx);
  sycl::free(C, ctx);

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

template <typename AccessorTy>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(AccessorTy acc, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE<AccessorTy>(acc, b, i);
  return res;
}
