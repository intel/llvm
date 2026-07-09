#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

inline void enqueueWithKernelId(sycl::queue &Q, sycl::kernel_id Id, int *Ptr) {
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context(),
                                                              {Q.get_device()});
  auto Kernel = KernelBundle.get_kernel(Id);
  Q.submit([&](sycl::handler &CGH) {
    CGH.set_args(Ptr);
    CGH.single_task(Kernel);
  });
}
