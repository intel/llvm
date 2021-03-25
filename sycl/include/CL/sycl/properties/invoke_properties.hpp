//==----------- kernel_properties.hpp --- SYCL kernel properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_helper.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {

enum gpu_cache_config { large_slm = 0, large_data = 1 };

namespace property {
namespace kernel {

class gpu_cache_config : public cl::sycl::detail::PropertyWithData<
                             cl::sycl::detail::GPUCacheConfig> {
public:
  gpu_cache_config(cl::sycl::INTEL::gpu_cache_config Config) : Config(Config) {}

  cl::sycl::INTEL::gpu_cache_config get_gpu_cache_config() const {
    return Config;
  }

private:
  cl::sycl::INTEL::gpu_cache_config Config;
};

} // namespace kernel
} // namespace property
} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
