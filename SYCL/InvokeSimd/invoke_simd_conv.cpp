// The test checks that invoke_simd implementation performs proper conversions
// on the actual arguments:
// - Case1: actual type is uniform<T>, formal - T1 (scalar)
//   standard C++ arithmetic conversion is applied
// - Case2: actual type is T, format - simd<T1, VL>
//   simd-simd conversion is applied according to the std::experimental::simd
//   specification. Basically, only non-narrowing conversions are allowed:
//   char -> int, float -> double, etc. int -> float is forbidden.

// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;

constexpr int VL = 16;

template <class T> inline T calc(T val) { return val + val; }
template <class T, int N> inline simd<T, N> calc(simd<T, N> val) {
  // emulate '+' on simd operands
  for (int i = 0; i < N; ++i) {
    val[i] += val[i];
  }
  return val;
}

template <class SimdElemT>
[[intel::device_indirectly_callable]] // required by FE for addr-taken functions
simd<SimdElemT, VL> __regcall SIMD_CALLEE_UNIFORM(SimdElemT val)
    SYCL_ESIMD_FUNCTION {
  return simd<SimdElemT, VL>(calc(val)); // broadcast
}

template <class SimdElemT>
[[intel::device_indirectly_callable]] simd<SimdElemT, VL> __regcall SIMD_CALLEE(
    simd<SimdElemT, VL> val) SYCL_ESIMD_FUNCTION {
  return calc(val);
}

class ESIMDSelector : public device_selector {
  // Require GPU device
  virtual int operator()(const device &device) const {
    if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      std::cerr
          << "Supported 'SYCL_DEVICE_FILTER' env var values is 'gpu' and  '"
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

template <class, class, bool> class TestID;

template <class SpmdT, class SimdElemT, bool IsUniform> bool test(queue q) {
  // 3 subgroups per workgroup
  unsigned GroupSize = VL * 3;
  unsigned NGroups = 7;
  unsigned Size = GroupSize * NGroups;
  SimdElemT *A = malloc_shared<SimdElemT>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = (SimdElemT)i;
  }
  sycl::range<1> GlobalRange{Size};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<SpmdT, SimdElemT, IsUniform>>(
          Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
            sub_group sg = ndi.get_sub_group();
            SpmdT val = (SpmdT)sg.get_group_linear_id(); // 0 .. GroupSize-1
            SimdElemT res = 0;

            if constexpr (IsUniform) {
              res =
                  invoke_simd(sg, SIMD_CALLEE_UNIFORM<SimdElemT>, uniform{val});
            } else {
              res = invoke_simd(sg, SIMD_CALLEE<SimdElemT>, val);
            }
            uint32_t i = ndi.get_global_linear_id();
            A[i] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    return false;
  }
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    unsigned group_id = i / GroupSize;
    uint32_t sg_id = (i - (group_id * GroupSize)) / VL;
    SimdElemT test = A[i];
    SimdElemT gold = calc((SimdElemT)sg_id);
    if ((test != gold) && (++err_cnt < 10)) {
      std::cout << "failed at index " << i << ", " << test << " != " << gold
                << "(gold)\n";
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }
  sycl::free(A, q);
  return err_cnt == 0;
}

int main(void) {
  queue q(ESIMDSelector{}, createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;

  constexpr bool UNIFORM = true;
  constexpr bool NON_UNIFORM = false;

  // With uniform parameters SPMD actual argument corresponds to SIMD scalar
  // argument, and standard C++ arithmetic conversion are implicitly
  // applied by the compiler. Any aritimetic type can be implicitly coverted to
  // any other arithmetic type.

  passed &= test<int, float, UNIFORM>(q);
  passed &= test<unsigned char, uint64_t, UNIFORM>(q);
  passed &= test<char, double, UNIFORM>(q);
  passed &= test<double, char, UNIFORM>(q);

  // With non-uniform parameters, SPMD actual argument of type T is "widened" to
  // std::simd<T, VL> and then convered to SIMD vector argument
  // (std::simd<T1, VL>) using std::simd implicit conversion constructors. They
  // allow only non-narrowing conversions (e.g. int -> float is narrowing and
  // hence is prohibited).

  passed &= test<char, long, NON_UNIFORM>(q);
  passed &= test<short, short, NON_UNIFORM>(q);
  passed &= test<float, double, NON_UNIFORM>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
