#pragma once

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))
#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]
#define __SYCL_BUILTIN_ALIAS(X) [[clang::builtin_alias(X)]]

#ifdef SYCL_EXTERNAL
#define __DPCPP_SYCL_EXTERNAL SYCL_EXTERNAL
#else
#ifdef __SYCL_DEVICE_ONLY__
#define __DPCPP_SYCL_EXTERNAL __attribute__((sycl_device))
#else
#define __DPCPP_SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif
#endif

// Sampler class without a default constructor.
namespace sycl {
inline namespace _V1 {
struct sampler_impl {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(sampler) sampler {
  struct sampler_impl impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
#endif

public:
  void use(void) const {}
};
} // namespace _V1
} // namespace sycl
