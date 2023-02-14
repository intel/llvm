//==------------------- device_aspect_macros.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_ALL_DEVICES_HAVE_0__
// __SYCL_ASPECT(host, 0)
#define __SYCL_ALL_DEVICES_HAVE_0__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_1__
// __SYCL_ASPECT(cpu, 1)
#define __SYCL_ALL_DEVICES_HAVE_1__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_2__
//__SYCL_ASPECT(gpu, 2)
#define __SYCL_ALL_DEVICES_HAVE_2__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_3__
//__SYCL_ASPECT(accelerator, 3)
#define __SYCL_ALL_DEVICES_HAVE_3__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_4__
//__SYCL_ASPECT(custom, 4)
#define __SYCL_ALL_DEVICES_HAVE_4__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_5__
// __SYCL_ASPECT(fp16, 5)
#define __SYCL_ALL_DEVICES_HAVE_5__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_6__
// __SYCL_ASPECT(fp64, 6)
#define __SYCL_ALL_DEVICES_HAVE_6__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_7__
// __SYCL_ASPECT_DEPRECATED(int64_base_atomics, 7)
#define __SYCL_ALL_DEVICES_HAVE_7__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_8__
// __SYCL_ASPECT_DEPRECATED(int64_extended_atomics, 8)
#define __SYCL_ALL_DEVICES_HAVE_8__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_9__
// __SYCL_ASPECT(image, 9)
#define __SYCL_ALL_DEVICES_HAVE_9__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_10__
// __SYCL_ASPECT(online_compiler, 10)
#define __SYCL_ALL_DEVICES_HAVE_10__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_11__
// __SYCL_ASPECT(online_linker, 11)
#define __SYCL_ALL_DEVICES_HAVE_11__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_12__
// __SYCL_ASPECT(queue_profiling, 12)
#define __SYCL_ALL_DEVICES_HAVE_12__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_13__
// __SYCL_ASPECT(usm_device_allocations, 13)
#define __SYCL_ALL_DEVICES_HAVE_13__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_14__
// __SYCL_ASPECT(usm_host_allocations, 14)
#define __SYCL_ALL_DEVICES_HAVE_14__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_15__
// __SYCL_ASPECT(usm_shared_allocations, 15)
#define __SYCL_ALL_DEVICES_HAVE_15__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_16__
// __SYCL_ASPECT(usm_system_allocations, 16)
#define __SYCL_ALL_DEVICES_HAVE_16__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_17__
// __SYCL_ASPECT(ext_intel_pci_address, 17)
#define __SYCL_ALL_DEVICES_HAVE_17__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_18__
// __SYCL_ASPECT(ext_intel_gpu_eu_count, 18)
#define __SYCL_ALL_DEVICES_HAVE_18__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_19__
// __SYCL_ASPECT(ext_intel_gpu_eu_simd_width, 19)
#define __SYCL_ALL_DEVICES_HAVE_19__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_20__
// __SYCL_ASPECT(ext_intel_gpu_slices, 20)
#define __SYCL_ALL_DEVICES_HAVE_20__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_21__
// __SYCL_ASPECT(ext_intel_gpu_subslices_per_slice, 21)
#define __SYCL_ALL_DEVICES_HAVE_21__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_22__
// __SYCL_ASPECT(ext_intel_gpu_eu_count_per_subslice, 22)
#define __SYCL_ALL_DEVICES_HAVE_22__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_23__
// __SYCL_ASPECT(ext_intel_max_mem_bandwidth, 23)
#define __SYCL_ALL_DEVICES_HAVE_23__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_24__
// __SYCL_ASPECT(ext_intel_mem_channel, 24)
#define __SYCL_ALL_DEVICES_HAVE_24__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_25__
// __SYCL_ASPECT(usm_atomic_host_allocations, 25)
#define __SYCL_ALL_DEVICES_HAVE_25__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_26__
// __SYCL_ASPECT(usm_atomic_shared_allocations, 26)
#define __SYCL_ALL_DEVICES_HAVE_26__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_27__
// __SYCL_ASPECT(atomic64, 27)
#define __SYCL_ALL_DEVICES_HAVE_27__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_28__
// __SYCL_ASPECT(ext_intel_device_info_uuid, 28)
#define __SYCL_ALL_DEVICES_HAVE_28__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_29__
// __SYCL_ASPECT(ext_oneapi_srgb, 29)
#define __SYCL_ALL_DEVICES_HAVE_29__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_30__
// __SYCL_ASPECT(ext_oneapi_native_assert, 30)
#define __SYCL_ALL_DEVICES_HAVE_30__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_31__
// __SYCL_ASPECT(host_debuggable, 31)
#define __SYCL_ALL_DEVICES_HAVE_31__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_32__
// __SYCL_ASPECT(ext_intel_gpu_hw_threads_per_eu, 32)
#define __SYCL_ALL_DEVICES_HAVE_32__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_33__
// __SYCL_ASPECT(ext_oneapi_cuda_async_barrier, 33)
#define __SYCL_ALL_DEVICES_HAVE_33__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_34__
// __SYCL_ASPECT(ext_oneapi_bfloat16_math_functions, 34)
#define __SYCL_ALL_DEVICES_HAVE_34__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_35__
// __SYCL_ASPECT(ext_intel_free_memory, 35)
#define __SYCL_ALL_DEVICES_HAVE_35__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_36__
// __SYCL_ASPECT(ext_intel_device_id, 36)
#define __SYCL_ALL_DEVICES_HAVE_36__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_37__
// __SYCL_ASPECT(ext_intel_memory_clock_rate, 37)
#define __SYCL_ALL_DEVICES_HAVE_37__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_38__
// __SYCL_ASPECT(ext_intel_memory_bus_width, 38)
#define __SYCL_ALL_DEVICES_HAVE_38__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_39__
// __SYCL_ASPECT(emulated, 39)
#define __SYCL_ALL_DEVICES_HAVE_39__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_0__
// __SYCL_ASPECT(host, 0)
#define __SYCL_ANY_DEVICE_HAS_0__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_1__
// __SYCL_ASPECT(cpu, 1)
#define __SYCL_ANY_DEVICE_HAS_1__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_2__
//__SYCL_ASPECT(gpu, 2)
#define __SYCL_ANY_DEVICE_HAS_2__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_3__
//__SYCL_ASPECT(accelerator, 3)
#define __SYCL_ANY_DEVICE_HAS_3__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_4__
//__SYCL_ASPECT(custom, 4)
#define __SYCL_ANY_DEVICE_HAS_4__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_5__
// __SYCL_ASPECT(fp16, 5)
#define __SYCL_ANY_DEVICE_HAS_5__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_6__
// __SYCL_ASPECT(fp64, 6)
#define __SYCL_ANY_DEVICE_HAS_6__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_7__
// __SYCL_ASPECT_DEPRECATED(int64_base_atomics, 7)
#define __SYCL_ANY_DEVICE_HAS_7__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_8__
// __SYCL_ASPECT_DEPRECATED(int64_extended_atomics, 8)
#define __SYCL_ANY_DEVICE_HAS_8__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_9__
// __SYCL_ASPECT(image, 9)
#define __SYCL_ANY_DEVICE_HAS_9__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_10__
// __SYCL_ASPECT(online_compiler, 10)
#define __SYCL_ANY_DEVICE_HAS_10__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_11__
// __SYCL_ASPECT(online_linker, 11)
#define __SYCL_ANY_DEVICE_HAS_11__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_12__
// __SYCL_ASPECT(queue_profiling, 12)
#define __SYCL_ANY_DEVICE_HAS_12__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_13__
// __SYCL_ASPECT(usm_device_allocations, 13)
#define __SYCL_ANY_DEVICE_HAS_13__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_14__
// __SYCL_ASPECT(usm_host_allocations, 14)
#define __SYCL_ANY_DEVICE_HAS_14__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_15__
// __SYCL_ASPECT(usm_shared_allocations, 15)
#define __SYCL_ANY_DEVICE_HAS_15__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_16__
// __SYCL_ASPECT(usm_system_allocations, 16)
#define __SYCL_ANY_DEVICE_HAS_16__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_17__
// __SYCL_ASPECT(ext_intel_pci_address, 17)
#define __SYCL_ANY_DEVICE_HAS_17__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_18__
// __SYCL_ASPECT(ext_intel_gpu_eu_count, 18)
#define __SYCL_ANY_DEVICE_HAS_18__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_19__
// __SYCL_ASPECT(ext_intel_gpu_eu_simd_width, 19)
#define __SYCL_ANY_DEVICE_HAS_19__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_20__
// __SYCL_ASPECT(ext_intel_gpu_slices, 20)
#define __SYCL_ANY_DEVICE_HAS_20__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_21__
// __SYCL_ASPECT(ext_intel_gpu_subslices_per_slice, 21)
#define __SYCL_ANY_DEVICE_HAS_21__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_22__
// __SYCL_ASPECT(ext_intel_gpu_eu_count_per_subslice, 22)
#define __SYCL_ANY_DEVICE_HAS_22__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_23__
// __SYCL_ASPECT(ext_intel_max_mem_bandwidth, 23)
#define __SYCL_ANY_DEVICE_HAS_23__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_24__
// __SYCL_ASPECT(ext_intel_mem_channel, 24)
#define __SYCL_ANY_DEVICE_HAS_24__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_25__
// __SYCL_ASPECT(usm_atomic_host_allocations, 25)
#define __SYCL_ANY_DEVICE_HAS_25__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_26__
// __SYCL_ASPECT(usm_atomic_shared_allocations, 26)
#define __SYCL_ANY_DEVICE_HAS_26__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_27__
// __SYCL_ASPECT(atomic64, 27)
#define __SYCL_ANY_DEVICE_HAS_27__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_28__
// __SYCL_ASPECT(ext_intel_device_info_uuid, 28)
#define __SYCL_ANY_DEVICE_HAS_28__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_29__
// __SYCL_ASPECT(ext_oneapi_srgb, 29)
#define __SYCL_ANY_DEVICE_HAS_29__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_30__
// __SYCL_ASPECT(ext_oneapi_native_assert, 30)
#define __SYCL_ANY_DEVICE_HAS_30__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_31__
// __SYCL_ASPECT(host_debuggable, 31)
#define __SYCL_ANY_DEVICE_HAS_31__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_32__
// __SYCL_ASPECT(ext_intel_gpu_hw_threads_per_eu, 32)
#define __SYCL_ANY_DEVICE_HAS_32__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_33__
// __SYCL_ASPECT(ext_oneapi_cuda_async_barrier, 33)
#define __SYCL_ANY_DEVICE_HAS_33__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_34__
// __SYCL_ASPECT(ext_oneapi_bfloat16_math_functions, 34)
#define __SYCL_ANY_DEVICE_HAS_34__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_35__
// __SYCL_ASPECT(ext_intel_free_memory, 35)
#define __SYCL_ANY_DEVICE_HAS_35__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_36__
// __SYCL_ASPECT(ext_intel_device_id, 36)
#define __SYCL_ANY_DEVICE_HAS_36__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_37__
// __SYCL_ASPECT(ext_intel_memory_clock_rate, 37)
#define __SYCL_ANY_DEVICE_HAS_37__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_38__
// __SYCL_ASPECT(ext_intel_memory_bus_width, 38)
#define __SYCL_ANY_DEVICE_HAS_38__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_39__
// __SYCL_ASPECT(emulated, 39)
#define __SYCL_ANY_DEVICE_HAS_39__ 0
#endif
