// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test checks that SYCL device functions cannot be called from ESIMD context.

__attribute__((sycl_device)) void sycl_func() {}
__attribute__((sycl_device)) void __spirv_reserved_func() {}
__attribute__((sycl_device)) void __sycl_reserved_func() {}
__attribute__((sycl_device)) void __other_reserved_func() {}

// -- Immediate diagnostic
__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func1() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  sycl_func();
  // Reserved SPIRV and SYCL functions are allowed
  __spirv_reserved_func();
  __sycl_reserved_func();
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  __other_reserved_func();
}

// -- Deferred diagnostic
void foo() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  sycl_func();
}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func2() {
  // expected-note@+1{{called by}}
  foo();
}

// -- Class method
struct S {
  __attribute__((sycl_device)) void sycl_func() {}
};

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func3() {
  S s;
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  s.sycl_func();
}

// -- Template function
template <typename Ty>
__attribute__((sycl_device)) void sycl_func() {}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func4() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  sycl_func<int>();
}

// -- std::function
namespace std {
template <typename _Tp>
_Tp declval();

template <typename _Functor, typename... _ArgTypes>
struct __res {
  template <typename... _Args>
  static decltype(declval<_Functor>()(_Args()...)) _S_test(int);

  template <typename...>
  static void _S_test(...);

  typedef decltype(_S_test<_ArgTypes...>(0)) type;
};

template <typename>
struct function;

template <typename _R, typename... _ArgTypes>
struct function<_R(_ArgTypes...)> {
  template <typename _Functor,
            typename = typename __res<_Functor, _ArgTypes...>::type>
  __attribute__((sycl_device, sycl_explicit_simd)) function(_Functor) {}
  __attribute__((sycl_device, sycl_explicit_simd)) _R operator()(_ArgTypes...) const;
};
} // namespace std

__attribute__((sycl_device)) void sycl_func1() {}

__attribute__((sycl_device, sycl_explicit_simd)) void passthrough(std::function<void(void)> &&C) { C(); }

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func5() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  passthrough(sycl_func1);
}
