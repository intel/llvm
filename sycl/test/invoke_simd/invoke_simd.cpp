// RUN: %clangxx -fsycl -fsyntax-only -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s
// FIXME: check if -fno-sycl-device-code-split-esimd affects any pre-link steps
//        and remove the flag if that is not the case

// The tests checks that invoke_simd API is compileable.

// TODO For now, compiling functors and lambdas as invoke_simd targets requires
// setting this macro before inclusion of invoke_simd.hpp macro. Remove when
// they are fully supported.
#define __INVOKE_SIMD_ENABLE_ALL_CALLABLES

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

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION;
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    void __regcall SIMD_CALLEE_VOID(simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {}

float SPMD_CALLEE(float *A, float b, int i) { return A[i] + b; }

class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in ONEAPI_DEVICE_SELECTOR env
  virtual int operator()(const device &device) const {
    if (const char *dev_filter = getenv("ONEAPI_DEVICE_SELECTOR")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      if (filter_string.find("host") != std::string::npos)
        return device.is_host() ? 1000 : -1;
      std::cerr
          << "Supported 'ONEAPI_DEVICE_SELECTOR' env var values are 'gpu' and "
             "'host', '"
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

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q(ESIMDSelector{}, createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
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

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i =
                sg.get_group_linear_id() * VL + g.get_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();
            float res = 0;

            if constexpr (use_invoke_simd) {
              res = invoke_simd(sg, SIMD_CALLEE, uniform{A}, B[wi_id],
                                uniform{i});
              invoke_simd(sg, SIMD_CALLEE_VOID, B[wi_id], uniform{i});
            } else {
              res = SPMD_CALLEE(A, B[wi_id], wi_id);
            }
            C[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
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

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res;
}

// Other test cases for compilation
// TODO convert to executable once lambdas and functors are supported in the
// middle-end.

// A functor with multiple '()' operator overloads.
struct SIMD_FUNCTOR {
  int Val;
  constexpr SIMD_FUNCTOR(int x) : Val(x) {}
  // annotated signature encoding wrt caller's return and argument types:
  // u - uniform, N - non-uniform, P - pointer (must be always uniform)

  // A - u(N, u)
  SYCL_EXTERNAL __regcall char operator()(simd<float, 16>, float) const;
  // B - u(N, u, u)
  SYCL_EXTERNAL __regcall int operator()(simd<float, 8>, float, int) const;
  // C - u(N, P)
  SYCL_EXTERNAL __regcall uniform<simd<float, 7>> operator()(simd<float, 16>,
                                                             float *) const;
  // D - u(P, u, u) - "all uniform", subgroup size does not matter
  SYCL_EXTERNAL __regcall uniform<simd<float, 8>>
  operator()(float *, simd<float, 3>, simd<int, 5>) const;
  // E - N(u, N)
  SYCL_EXTERNAL __regcall simd<short, 8> operator()(simd<float, 3>,
                                                    simd<int, 8>) const;
  // F - void
  SYCL_EXTERNAL __regcall void operator()(simd<float, 3>) const;
};

// Functor-based tests.
SYCL_EXTERNAL void foo(sub_group sg, float a, float b, float *ptr) {
  SIMD_FUNCTOR ftor{10};
  // the target is "A" SIMD_FUNCTOR::() overload:
  auto x = invoke_simd(sg, ftor, 1.f, uniform{a});
  static_assert(std::is_same_v<decltype(x), uniform<char>>);

  // the target is "B" SIMD_FUNCTOR::() overload:
  auto y = invoke_simd(sg, ftor, b, uniform{1.f}, uniform{10});
  static_assert(std::is_same_v<decltype(y), uniform<int>>);

  // the target is "C" SIMD_FUNCTOR::() overload:
  auto z = invoke_simd(sg, ftor, b, uniform{ptr});
  static_assert(std::is_same_v<decltype(z), uniform<simd<float, 7>>>);

  // the target is "D" SIMD_FUNCTOR::() overload:
  auto u = invoke_simd(sg, ftor, uniform{ptr}, uniform{simd<float, 3>{1}},
                       uniform{simd<int, 5>{2}});
  static_assert(std::is_same_v<decltype(u), uniform<simd<float, 8>>>);

  // the target is "E" SIMD_FUNCTOR::() overload:
  auto v = invoke_simd(sg, ftor, uniform{simd<float, 3>{1}}, 1);
  static_assert(std::is_same_v<decltype(v), short>);

  // the target is "F" SIMD_FUNCTOR::() overload:
  invoke_simd(sg, ftor, uniform{simd<float, 3>{1}});

}

// Lambda-based tests, repeat functor test cases above.
SYCL_EXTERNAL auto bar(sub_group sg, float a, float b, float *ptr, char ch) {
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 16>, float) {
      // capturing lambda
      return ch;
    };
    auto x = invoke_simd(sg, ftor, 1.f, uniform{a});
    static_assert(std::is_same_v<decltype(x), uniform<char>>);
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 8>, float, int) {
      // non-capturing lambda
      return (int)10;
    };
    auto y = invoke_simd(sg, ftor, b, uniform{1.f}, uniform{10});
    static_assert(std::is_same_v<decltype(y), uniform<int>>);
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 16>, float *) {
      simd<float, 7> val{ch};
      return uniform{val};
    };
    auto z = invoke_simd(sg, ftor, b, uniform{ptr});
    static_assert(std::is_same_v<decltype(z), uniform<simd<float, 7>>>);
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (float *, simd<float, 3>,
                                            simd<int, 5>) {
      simd<float, 8> val{ch};
      return uniform{val};
    };
    auto u = invoke_simd(sg, ftor, uniform{ptr}, uniform{simd<float, 3>{1}},
                         uniform{simd<int, 5>{2}});
    static_assert(std::is_same_v<decltype(u), uniform<simd<float, 8>>>);
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 3>, simd<int, 8>) {
      return simd<short, 8>{};
    };
    auto v = invoke_simd(sg, ftor, uniform{simd<float, 3>{1}}, 1);
    static_assert(std::is_same_v<decltype(v), short>);
  }

  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 16>, float) {};
    invoke_simd(sg, ftor, 1.f, uniform{a});
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 8>, float, int) {};
    invoke_simd(sg, ftor, b, uniform{1.f}, uniform{10});
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 16>, float *) {};
    invoke_simd(sg, ftor, b, uniform{ptr});
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (float *, simd<float, 3>,
                                            simd<int, 5>) {};
    invoke_simd(sg, ftor, uniform{ptr}, uniform{simd<float, 3>{1}},
                         uniform{simd<int, 5>{2}});
  }
  {
    const auto ftor = [=] [[gnu::regcall]] (simd<float, 3>, simd<int, 8>) {};
    invoke_simd(sg, ftor, uniform{simd<float, 3>{1}}, 1);
  }
}

// Function-pointer-based test
SYCL_EXTERNAL auto barx(sub_group sg, float a, char ch,
                        __regcall char(f)(simd<float, 16>, float)) {
  auto x = invoke_simd(sg, f, 1.f, uniform{a});
  static_assert(std::is_same_v<decltype(x), uniform<char>>);
}

SYCL_EXTERNAL auto barx_void(sub_group sg, float a, char ch,
                        __regcall void(f)(simd<float, 16>, float)) {
  invoke_simd(sg, f, 1.f, uniform{a});
}

// Internal is_function_ref_v meta-API checks {
template <class F> void assert_is_func(F &&f) {
  static_assert(
      sycl::ext::oneapi::experimental::detail::is_function_ptr_or_ref_v<F>);
}

template <class F> void assert_is_not_func(F &&f) {
  static_assert(
      !sycl::ext::oneapi::experimental::detail::is_function_ptr_or_ref_v<F>);
}

void ordinary_func();

// clang-format off
void check_f(
  int(*func_ptr)(float*), int(__regcall* func_ptr_regcall)(float*, int),
  int(&func_ref)(), int(__regcall& func_ref_regcall)(int*),
  int(func)(float), int(__regcall func_regcall)(int)) {

  assert_is_func(SIMD_CALLEE);
  assert_is_func(SIMD_CALLEE_VOID);
  assert_is_func(ordinary_func);

  assert_is_func(func_ptr);
  assert_is_func(func_ptr_regcall);

  assert_is_func(func_ref);
  assert_is_func(func_ref_regcall);

  assert_is_func(func);
  assert_is_func(func_regcall);
}
// clang-format on

void check_not_f(char ch) {
  assert_is_not_func(SIMD_FUNCTOR{10});
  const auto capt_lambda = [=] [[gnu::regcall]] (simd<float, 16>, float) {
    // capturing lambda
    return ch;
  };
  const auto non_capt_lambda = [](simd<float, 16>, float) {
    // non-capturing lambda
    return 10;
  };
  assert_is_not_func(capt_lambda);
  assert_is_not_func(non_capt_lambda);
}
