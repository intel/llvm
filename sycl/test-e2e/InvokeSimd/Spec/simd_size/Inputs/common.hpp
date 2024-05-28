#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/usm.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

template <int Size, int VL>
ESIMD_NOINLINE esimd::simd<float, Size>
ESIMD_CALLEE(float *A, esimd::simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {
  constexpr int STEP = Size > VL ? VL : Size;
  constexpr int ROWS = Size / STEP;

  esimd::simd<float, Size> res(0);
  for (int j = 0; j < ROWS; j++) {
    esimd::simd<float, STEP> a;
    a.copy_from(A + i + j * STEP);

    if (j % 2 == 0)
      res.template select<STEP, 1>(j * STEP) =
          a + b.template select<STEP, 1>(j * STEP);
    else
      res.template select<STEP, 1>(j * STEP) =
          i * a - b.template select<STEP, 1>(j * STEP);
  }
  return res;
}

template <int Size, int VL>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, Size> __regcall SIMD_CALLEE(float *A, simd<float, VL> b,
                                            int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, Size> res = ESIMD_CALLEE<Size, VL>(A, b, i);
  return res;
}

using namespace sycl;

template <int, int> class TestID;

template <int Size, int VL, class QueueTY> bool test(QueueTY q) {
  std::cout << "Case: [" << Size << ", " << VL << "]\n";

  constexpr int GroupSize = 1;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

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

  sycl::range<1> GroupRange{Size};
  sycl::range<1> TaskRange{GroupSize};
  sycl::nd_range<1> Range(GroupRange, TaskRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class TestID<Size, VL>>(
          Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i = sg.get_group_linear_id() * VL +
                         g.get_group_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();

            float res = invoke_simd(sg, SIMD_CALLEE<Size, VL>, uniform{A},
                                    B[wi_id], uniform{i});
            C[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, ctxt);
    sycl::free(B, ctxt);
    sycl::free(C, ctxt);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  int err_cnt = 0;

  constexpr int STEP = Size > VL ? VL : Size;
  constexpr int ROWS = Size / STEP;

  for (int j = 0; j < ROWS; j++) {
    for (unsigned i = 0; i < STEP; ++i) {
      int k = j * STEP + i;

      if (j % 2 == 0) {
        if ((A[k] + B[k]) != C[k]) {
          err_cnt++;
          std::cout << "failed at index " << k << ", " << C[k] << " != " << A[k]
                    << " + " << B[k] << "\n";
        }
      } else {
        if ((i * A[k] - B[k]) != C[k]) {
          err_cnt++;
          std::cout << "failed at index " << k << ", " << C[k] << " != 3 * "
                    << A[k] << " - " << B[k] << "\n";
        }
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  sycl::free(A, ctxt);
  sycl::free(B, ctxt);
  sycl::free(C, ctxt);

  return err_cnt == 0;
}
