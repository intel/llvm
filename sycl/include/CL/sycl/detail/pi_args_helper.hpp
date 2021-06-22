//==---------- pi_args_helper.hpp - PI call arguments helper ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_args_helper.hpp
/// Utilities to enable PI call arguments packing for XPTI.
///
/// C++ wrapper for PI does not take real function argument type into account.
/// As a result, when Plugin.call<>() is invoked, there might be type mismatch
/// between deduced type and real call argument type (e.g. when there's
/// std::vector::size() or an integer literal in call expression). This leads to
/// unstable data exchange format between SYCL runtime and XPTI. To workaround
/// the problem, SYCL runtime must explicitly cast template-deduced types to
/// real types before packing arguments for XPTI. This file contains mappings
/// between PiApiKind and tuples of argument types.
///
/// \ingroup sycl_pi

#pragma once

#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>

#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <PiApiKind ApiKind> struct PiApiArgTuple;

template <> struct PiApiArgTuple<detail::PiApiKind::piPlatformsGet> {
  using type = std::tuple<pi_uint32, pi_platform *, pi_uint32 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piPlatformGetInfo> {
  using type =
      std::tuple<pi_platform, pi_platform_info, size_t, void *, size_t *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextPlatformGetNativeHandle> {
  using type = std::tuple<pi_platform, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextPlatformCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_platform *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDevicesGet> {
  using type = std::tuple<pi_platform, pi_device_type, pi_uint32, pi_device *,
                          pi_uint32 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceGetInfo> {
  using type = std::tuple<pi_device, pi_device_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceRetain> {
  using type = std::tuple<pi_device>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceRelease> {
  using type = std::tuple<pi_device>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDevicePartition> {
  using type = std::tuple<pi_device, const pi_device_partition_property *,
                          pi_uint32, pi_device *, pi_uint32 *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextDeviceGetNativeHandle> {
  using type = std::tuple<pi_device, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextDeviceCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_platform, pi_device *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextDeviceSelectBinary> {
  using type =
      std::tuple<pi_device, pi_device_binary *, pi_uint32, pi_uint32 *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextGetDeviceFunctionPointer> {
  using type = std::tuple<pi_device, pi_program, const char *, pi_uint64 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextCreate> {
  using type =
      std::tuple<const pi_context_properties *, pi_uint32, const pi_device *,
                 void (*)(const char *, const void *, size_t, void *), void *,
                 pi_context *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextGetInfo> {
  using type =
      std::tuple<pi_context, pi_context_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextRetain> {
  using type = std::tuple<pi_context>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextRelease> {
  using type = std::tuple<pi_context>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextSetExtendedDeleter> {
  using type = std::tuple<pi_context, pi_context_extended_deleter, void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextGetNativeHandle> {
  using type = std::tuple<pi_context, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_uint32, const pi_device *, bool,
                          pi_context *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueCreate> {
  using type =
      std::tuple<pi_context, pi_device, pi_queue_properties, pi_queue *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueGetInfo> {
  using type = std::tuple<pi_queue, pi_queue_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueRetain> {
  using type = std::tuple<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueRelease> {
  using type = std::tuple<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueFinish> {
  using type = std::tuple<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextQueueGetNativeHandle> {
  using type = std::tuple<pi_queue, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextQueueCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_context, pi_queue *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemBufferCreate> {
  using type = std::tuple<pi_context, pi_mem_flags, size_t, void *, pi_mem *,
                          const pi_mem_properties *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemImageCreate> {
  using type = std::tuple<pi_context, pi_mem_flags, const pi_image_format *,
                          const pi_image_desc *, void *, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemGetInfo> {
  using type = std::tuple<pi_mem, cl_mem_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemImageGetInfo> {
  using type = std::tuple<pi_mem, pi_image_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemRetain> {
  using type = std::tuple<pi_mem>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemRelease> {
  using type = std::tuple<pi_mem>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemBufferPartition> {
  using type =
      std::tuple<pi_mem, pi_mem_flags, pi_buffer_create_type, void *, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextMemGetNativeHandle> {
  using type = std::tuple<pi_mem, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextMemCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCreate> {
  using type = std::tuple<pi_context, const void *, size_t, pi_program *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piclProgramCreateWithSource> {
  using type = std::tuple<pi_context, pi_uint32, const char **, const size_t,
                          pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCreateWithBinary> {
  using type =
      std::tuple<pi_context, pi_uint32, const pi_device *, const size_t *,
                 const unsigned char **, pi_uint32 *, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramGetInfo> {
  using type =
      std::tuple<pi_program, pi_program_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramLink> {
  using type = std::tuple<pi_context, pi_uint32, const pi_device *,
                          const char *, pi_uint32, const pi_program *,
                          void (*)(pi_program, void *), void *, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCompile> {
  using type = std::tuple<pi_program, pi_uint32, const pi_device *,
                          const char *, pi_uint32, const pi_program *,
                          const char **, void (*)(pi_program, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramBuild> {
  using type = std::tuple<pi_program, pi_uint32, const pi_device *,
                          const char *, void (*)(pi_program, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramGetBuildInfo> {
  using type = std::tuple<pi_program, pi_device, cl_program_build_info, size_t,
                          void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramRetain> {
  using type = std::tuple<pi_program>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramRelease> {
  using type = std::tuple<pi_program>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramSetSpecializationConstant> {
  using type = std::tuple<pi_program, pi_uint32, size_t, const void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramGetNativeHandle> {
  using type = std::tuple<pi_program, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_context, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelCreate> {
  using type = std::tuple<pi_program, const char *, pi_kernel *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelSetArg> {
  using type = std::tuple<pi_kernel, pi_uint32, size_t, const void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetInfo> {
  using type = std::tuple<pi_kernel, pi_kernel_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetGroupInfo> {
  using type = std::tuple<pi_kernel, pi_device, pi_kernel_group_info, size_t,
                          void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetSubGroupInfo> {
  using type = std::tuple<pi_kernel, pi_device, pi_kernel_sub_group_info,
                          size_t, const void *, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelRetain> {
  using type = std::tuple<pi_kernel>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelRelease> {
  using type = std::tuple<pi_kernel>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgPointer> {
  using type = std::tuple<pi_kernel, pi_uint32, size_t, const void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelSetExecInfo> {
  using type = std::tuple<pi_kernel, pi_kernel_exec_info, size_t, const void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextKernelCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_context, bool, pi_kernel *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextKernelGetNativeHandle> {
  using type = std::tuple<pi_kernel, pi_native_handle *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventCreate> {
  using type = std::tuple<pi_context, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventGetInfo> {
  using type = std::tuple<pi_event, pi_event_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventGetProfilingInfo> {
  using type =
      std::tuple<pi_event, pi_profiling_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventsWait> {
  using type = std::tuple<pi_uint32, const pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventSetCallback> {
  using type = std::tuple<pi_event, pi_int32,
                          void (*)(pi_event, pi_int32, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventSetStatus> {
  using type = std::tuple<pi_event, pi_int32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventRetain> {
  using type = std::tuple<pi_event>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventRelease> {
  using type = std::tuple<pi_event>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextEventGetNativeHandle> {
  using type = std::tuple<pi_event, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextEventCreateWithNativeHandle> {
  using type = std::tuple<pi_native_handle, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerCreate> {
  using type =
      std::tuple<pi_context, const pi_sampler_properties *, pi_sampler *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerGetInfo> {
  using type =
      std::tuple<pi_sampler, pi_sampler_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerRetain> {
  using type = std::tuple<pi_sampler>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerRelease> {
  using type = std::tuple<pi_sampler>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueKernelLaunch> {
  using type =
      std::tuple<pi_queue, pi_kernel, pi_uint32, const size_t *, const size_t *,
                 const size_t *, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueNativeKernel> {
  using type = std::tuple<pi_queue, void (*)(void *), void *, size_t, pi_uint32,
                          const pi_mem *, const void **, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueEventsWait> {
  using type = std::tuple<pi_queue, pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueEventsWaitWithBarrier> {
  using type = std::tuple<pi_queue, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferRead> {
  using type = std::tuple<pi_queue, pi_mem, pi_bool, size_t, size_t, void *,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferReadRect> {
  using type = std::tuple<pi_queue, pi_mem, pi_bool, pi_buff_rect_offset,
                          pi_buff_rect_offset, pi_buff_rect_region, size_t,
                          size_t, size_t, size_t, void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferWrite> {
  using type =
      std::tuple<pi_queue, pi_mem, pi_bool, size_t, size_t, const void *,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferWriteRect> {
  using type = std::tuple<pi_queue, pi_mem, pi_bool, pi_buff_rect_offset,
                          pi_buff_rect_offset, pi_buff_rect_region, size_t,
                          size_t, size_t, size_t, const void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferCopy> {
  using type = std::tuple<pi_queue, pi_mem, pi_mem, size_t, size_t, size_t,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferCopyRect> {
  using type =
      std::tuple<pi_queue, pi_mem, pi_mem, pi_buff_rect_offset,
                 pi_buff_rect_offset, pi_buff_rect_region, size_t, size_t,
                 size_t, size_t, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferFill> {
  using type = std::tuple<pi_queue, pi_mem, const void *, size_t, size_t,
                          size_t, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageRead> {
  using type = std::tuple<pi_queue, pi_mem, pi_bool, pi_image_offset,
                          pi_image_region, size_t, size_t, void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageWrite> {
  using type = std::tuple<pi_queue, pi_mem, pi_bool, pi_image_offset,
                          pi_image_region, size_t, size_t, const void *,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageCopy> {
  using type =
      std::tuple<pi_queue, pi_mem, pi_mem, pi_image_offset, pi_image_offset,
                 pi_image_region, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageFill> {
  using type =
      std::tuple<pi_queue, pi_mem, const void *, const size_t *, const size_t *,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferMap> {
  using type =
      std::tuple<pi_queue, pi_mem, pi_bool, pi_map_flags, size_t, size_t,
                 pi_uint32, const pi_event *, pi_event *, void **>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemUnmap> {
  using type = std::tuple<pi_queue, pi_mem, void *, pi_uint32, const pi_event *,
                          pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgMemObj> {
  using type = std::tuple<pi_kernel, pi_uint32, const pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgSampler> {
  using type = std::tuple<pi_kernel, pi_uint32, const pi_sampler *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMHostAlloc> {
  using type = std::tuple<void **, pi_context, pi_usm_mem_properties *, size_t,
                          pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMDeviceAlloc> {
  using type = std::tuple<void **, pi_context, pi_device,
                          pi_usm_mem_properties *, size_t, pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMSharedAlloc> {
  using type = std::tuple<void **, pi_context, pi_device,
                          pi_usm_mem_properties *, size_t, pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMFree> {
  using type = std::tuple<pi_context, void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemset> {
  using type = std::tuple<pi_queue, void *, pi_int32, size_t, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemcpy> {
  using type = std::tuple<pi_queue, pi_bool, void *, const void *, size_t,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueuePrefetch> {
  using type =
      std::tuple<pi_queue, const void *, size_t, pi_usm_migration_flags,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemAdvise> {
  using type =
      std::tuple<pi_queue, const void *, size_t, pi_mem_advice, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMGetMemAllocInfo> {
  using type = std::tuple<pi_context, const void *, pi_mem_info, size_t, void *,
                          size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextPluginGetOpaqueData> {
  using type = std::tuple<void *, void **>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piTearDown> {
  using type = std::tuple<void *>;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
