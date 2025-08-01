//==------------ sycl.hpp - SYCL2020 standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Throw warning when including sycl.hpp without using -fsycl flag.
// Warning can be disabled by defining SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro.
#define __SYCL_STRINGIFY(x) #x
#define __SYCL_TOSTRING(x) __SYCL_STRINGIFY(x)

#ifdef _MSC_VER
#define __SYCL_WARNING(msg)                                                    \
  __pragma(message(__FILE__ "(" __SYCL_TOSTRING(__LINE__) "): warning: " msg))
#elif defined(__GNUC__) || defined(__clang__)
#define __SYCL_WARNING(msg) _Pragma(__SYCL_TOSTRING(GCC warning msg))
#else
#define __SYCL_WARNING(msg) // Unsupported compiler
#endif

#if !defined(SYCL_LANGUAGE_VERSION) &&                                         \
    !defined(SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)
__SYCL_WARNING("You are including <sycl/sycl.hpp> without -fsycl flag, \
which is errorenous for device code compilation. This warning \
can be disabled by setting SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro.")
#endif
#undef __SYCL_WARNING
#undef __SYCL_TOSTRING
#undef __SYCL_STRINGIFY

#include <sycl/detail/core.hpp>

#include <sycl/accessor_image.hpp>
#include <sycl/aspects.hpp>
#include <sycl/atomic.hpp>
#include <sycl/atomic_fence.hpp>
#include <sycl/atomic_ref.hpp>
#include <sycl/backend.hpp>
#if SYCL_BACKEND_OPENCL
#include <sycl/backend/opencl.hpp>
#endif
#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif
#include <sycl/builtins.hpp>
#include <sycl/context.hpp>
#include <sycl/define_vendors.hpp>
#include <sycl/detail/vector_convert.hpp>
#include <sycl/device.hpp>
#include <sycl/device_aspect_traits.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/functional.hpp>
#include <sycl/group.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/handler.hpp>
#include <sycl/id.hpp>
#include <sycl/image.hpp>
#include <sycl/interop_handle.hpp>
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
#include <sycl/range.hpp>
#include <sycl/reduction.hpp>
#include <sycl/sampler.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/stream.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <sycl/usm/usm_pointer_info.hpp>
#include <sycl/vector.hpp>
#include <sycl/version.hpp>

#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/ext/intel/experimental/fp_control_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_datapath.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_mem.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp>
#include <sycl/ext/intel/experimental/pipe_properties.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/experimental/task_sequence_properties.hpp>
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/ext/intel/usm_pointers.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
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
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/ext/oneapi/experimental/chunk.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/composite_device.hpp>
#include <sycl/ext/oneapi/experimental/cuda/barrier.hpp>
#include <sycl/ext/oneapi/experimental/cuda/non_uniform_algorithms.hpp>
#include <sycl/ext/oneapi/experimental/current_device.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/event_mode_property.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/group_helpers_sorters.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/experimental/group_sort.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/experimental/reduction_properties.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/ext/oneapi/experimental/tangle.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/ext/oneapi/owner_less.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <sycl/ext/oneapi/sub_group.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <sycl/ext/oneapi/weak_object.hpp>
#include <sycl/khr/dynamic_addrspace_cast.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/khr/group_interface.hpp>
#include <sycl/khr/static_addrspace_cast.hpp>
#include <sycl/khr/work_item_queries.hpp>
