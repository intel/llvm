//==------------ sycl.hpp - SYCL2020 standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/aspects.hpp>
#include <sycl/atomic.hpp>
#include <sycl/atomic_fence.hpp>
#include <sycl/atomic_ref.hpp>
#include <sycl/backend.hpp>
#if SYCL_BACKEND_OPENCL
#include <sycl/backend/opencl.hpp>
#endif
#include <sycl/buffer.hpp>
#include <sycl/builtins.hpp>
#include <sycl/context.hpp>
#include <sycl/define_vendors.hpp>
#include <sycl/device.hpp>
#include <sycl/device_aspect_traits.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/experimental/group_sort.hpp>
#include <sycl/feature_test.hpp>
#include <sycl/functional.hpp>
#include <sycl/group.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/handler.hpp>
#include <sycl/id.hpp>
#include <sycl/image.hpp>
#include <sycl/item.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_handler.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/pipes.hpp>
#include <sycl/platform.hpp>
#include <sycl/pointers.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>
#include <sycl/reduction.hpp>
#include <sycl/sampler.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/stream.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/types.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <sycl/usm/usm_pointer_info.hpp>
#include <sycl/version.hpp>
#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/ext/intel/experimental/fp_control_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_datapath.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_mem.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp>
#include <sycl/ext/intel/experimental/pipe_properties.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/ext/intel/usm_pointers.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <sycl/ext/oneapi/device_global/properties.hpp>
#include <sycl/ext/oneapi/experimental/address_cast.hpp>
#include <sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_device.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_host.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_shared.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/dealloc.hpp>
#include <sycl/ext/oneapi/experimental/auto_local_range.hpp>
#include <sycl/ext/oneapi/experimental/ballot_group.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/ext/oneapi/experimental/cuda/barrier.hpp>
#include <sycl/ext/oneapi/experimental/fixed_size_group.hpp>
#include <sycl/ext/oneapi/experimental/opportunistic_group.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/ext/oneapi/experimental/tangle_group.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/owner_less.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <sycl/ext/oneapi/sub_group.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/ext/oneapi/weak_object.hpp>
#if !defined(SYCL2020_CONFORMANT_APIS) &&                                      \
    !defined(__INTEL_PREVIEW_BREAKING_CHANGES)
// We used to include those and some code might be reliant on that.
#include <cmath>
#include <complex>
#endif
