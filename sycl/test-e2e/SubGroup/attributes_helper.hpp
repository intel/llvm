//==- attributes_helper.hpp - SYCL sub_group attributes helper -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"

#define KERNEL_FUNCTOR_WITH_SIZE(SIZE)                                         \
  class KernelFunctor##SIZE {                                                  \
  public:                                                                      \
    [[sycl::reqd_sub_group_size(SIZE)]] void                                   \
    operator()(sycl::nd_item<1> Item) const {                                  \
      const auto GID = Item.get_global_id();                                   \
    }                                                                          \
  };

// Dummy kernel, so we get the types and can keep later code straight-lined.
#define DUMMY_KERNEL_FUNCTOR(SIZE)                                             \
  class KernelFunctor##SIZE {                                                  \
  public:                                                                      \
    void operator()(sycl::nd_item<1> Item) const {                             \
      const auto GID = Item.get_global_id();                                   \
    }                                                                          \
  };

#ifdef BUILD_FOR_GPU
DUMMY_KERNEL_FUNCTOR(1);
DUMMY_KERNEL_FUNCTOR(2);
DUMMY_KERNEL_FUNCTOR(4);
DUMMY_KERNEL_FUNCTOR(8);
DUMMY_KERNEL_FUNCTOR(16);
KERNEL_FUNCTOR_WITH_SIZE(32);
DUMMY_KERNEL_FUNCTOR(64);
#else
KERNEL_FUNCTOR_WITH_SIZE(1);
KERNEL_FUNCTOR_WITH_SIZE(2);
KERNEL_FUNCTOR_WITH_SIZE(4);
KERNEL_FUNCTOR_WITH_SIZE(8);
KERNEL_FUNCTOR_WITH_SIZE(16);
KERNEL_FUNCTOR_WITH_SIZE(32);
KERNEL_FUNCTOR_WITH_SIZE(64);
#endif

#undef KERNEL_FUNCTOR_WITH_SIZE

inline uint32_t flp2(uint32_t X) {
  X = X | (X >> 1);
  X = X | (X >> 2);
  X = X | (X >> 4);
  X = X | (X >> 8);
  X = X | (X >> 16);
  return X - (X >> 1);
}

template <typename Fn> inline void submit(sycl::queue &Q) {
  Q.submit([](sycl::handler &cgh) {
    Fn F;
    cgh.parallel_for(sycl::nd_range<1>{64, 16}, F);
  });
}

int runTests() {
  queue Queue;
  device Device = Queue.get_device();

  try {
    const auto SGSizes = Device.get_info<info::device::sub_group_sizes>();

    for (const auto SGSize : SGSizes) {
      // Get the previous power of 2
      auto ReqdSize = flp2(SGSize);

      std::cout << "Run for " << ReqdSize << " required workgroup size.\n";

      // Store the `sycl::kernel` into a vector because `sycl::kernel`
      // doesn't have default constructor
      std::vector<sycl::kernel> TheKernel;

      switch (ReqdSize) {
      case 64: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor64>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor64>(Queue);
        break;
      }
      case 32: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor32>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor32>(Queue);
        break;
      }
      case 16: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor16>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor16>(Queue);
        break;
      }
      case 8: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor8>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor8>(Queue);
        break;
      }
      case 4: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor4>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor4>(Queue);
        break;
      }
      case 2: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor2>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor2>(Queue);
        break;
      }
      case 1: {
        auto KernelID = sycl::get_kernel_id<KernelFunctor1>();
        auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Queue.get_context(), {KernelID});
        TheKernel.push_back(KB.get_kernel(KernelID));
        submit<KernelFunctor1>(Queue);
        break;
      }
      default:
        throw sycl::exception(sycl::errc::feature_not_supported,
                              "sub-group size is not supported");
      }

      auto Kernel = TheKernel[0];

      auto Res = Kernel.get_info<
          sycl::info::kernel_device_specific::compile_sub_group_size>(Device);

#ifdef BUILD_FOR_GPU
      // GPU targets only test this one size, override the value, so the check
      // passes and the code path don't diverge.
      if (ReqdSize != 32)
        ReqdSize = 0;
#endif

      exit_if_not_equal<size_t>(Res, ReqdSize, "compile_sub_group_size");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
