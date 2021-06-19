#pragma once

#include "tuple_view.hpp"
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace xpti_helpers {
template <detail::PiApiKind ApiKind> struct PiApiArgTuple;

template <> struct PiApiArgTuple<detail::PiApiKind::piPlatformsGet> {
  using type = tuple_view<pi_uint32, pi_platform *, pi_uint32 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piPlatformGetInfo> {
  using type =
      tuple_view<pi_platform, pi_platform_info, size_t, void *, size_t>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextPlatformGetNativeHandle> {
  using type = tuple_view<pi_platform, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextPlatformCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_platform *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDevicesGet> {
  using type = tuple_view<pi_platform, pi_device_type, pi_uint32, pi_device *,
                          pi_uint32 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceGetInfo> {
  using type = tuple_view<pi_device, pi_device_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceRetain> {
  using type = tuple_view<pi_device>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDeviceRelease> {
  using type = tuple_view<pi_device>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piDevicePartition> {
  using type = tuple_view<pi_device, const pi_device_partition_property *,
                          pi_uint32, pi_device, pi_uint32>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextDeviceGetNativeHandle> {
  using type = tuple_view<pi_device, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextDeviceCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_device *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextDeviceSelectBinary> {
  using type =
      tuple_view<pi_device, pi_device_binary *, pi_uint32, pi_uint32 *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextGetDeviceFunctionPointer> {
  using type = tuple_view<pi_device, pi_program, const char *, pi_uint64 *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextCreate> {
  using type =
      tuple_view<const pi_context_properties *, pi_uint32, const pi_device *,
                 void (*)(const char *, const void *, size_t, void *)>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextGetInfo> {
  using type =
      tuple_view<pi_context, pi_context_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextRetain> {
  using type = tuple_view<pi_context>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piContextRelease> {
  using type = tuple_view<pi_context>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextSetExtendedDeleter> {
  using type = tuple_view<pi_context, pi_context_extended_deleter, void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextGetNativeHandle> {
  using type = tuple_view<pi_context, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextContextCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_uint32, const pi_device *, bool,
                          pi_context *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueCreate> {
  using type =
      tuple_view<pi_context, pi_device, pi_queue_properties, pi_queue *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueGetInfo> {
  using type = tuple_view<pi_queue, pi_queue_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueRetain> {
  using type = tuple_view<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueRelease> {
  using type = tuple_view<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piQueueFinish> {
  using type = tuple_view<pi_queue>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextQueueGetNativeHandle> {
  using type = tuple_view<pi_queue, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextQueueCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_context, pi_queue *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemBufferCreate> {
  using type = tuple_view<pi_context, pi_mem_flags, size_t, void *, pi_mem *,
                          const pi_mem_properties *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemImageCreate> {
  using type = tuple_view<pi_context, pi_mem_flags, const pi_image_format *,
                          const pi_image_desc *, void *, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemGetInfo> {
  using type = tuple_view<pi_mem, cl_mem_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemImageGetInfo> {
  using type = tuple_view<pi_mem, pi_image_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemRetain> {
  using type = tuple_view<pi_mem>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemRelease> {
  using type = tuple_view<pi_mem>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piMemBufferPartition> {
  using type =
      tuple_view<pi_mem, pi_mem_flags, pi_buffer_create_type, void *, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextMemGetNativeHandle> {
  using type = tuple_view<pi_mem, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextMemCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCreate> {
  using type = tuple_view<pi_context, const void *, size_t, pi_program *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piclProgramCreateWithSource> {
  using type = tuple_view<pi_context, pi_uint32, const char **, const size_t,
                          pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCreateWithBinary> {
  using type =
      tuple_view<pi_context, pi_uint32, const pi_device *, const size_t *,
                 const unsigned char **, pi_uint32 *, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramGetInfo> {
  using type =
      tuple_view<pi_program, pi_program_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramLink> {
  using type = tuple_view<pi_context, pi_uint32, const char *, pi_uint32,
                          const pi_program *, void (*)(pi_program, void *),
                          void *, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramCompile> {
  using type = tuple_view<pi_program, pi_uint32, const pi_device *,
                          const char *, pi_uint32, const pi_program *,
                          const char **, void (*)(pi_program, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramBuild> {
  using type = tuple_view<pi_program, pi_uint32, const pi_device *,
                          const char *, void (*)(pi_program, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramGetBuildInfo> {
  using type = tuple_view<pi_program, pi_device, cl_program_build_info, size_t,
                          void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramRetain> {
  using type = tuple_view<pi_program>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piProgramRelease> {
  using type = tuple_view<pi_program>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramSetSpecializationConstant> {
  using type = tuple_view<pi_program, pi_uint32, size_t, const void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramGetNativeHandle> {
  using type = tuple_view<pi_program, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextProgramCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_program *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelCreate> {
  using type = tuple_view<pi_program, const char *, pi_kernel *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelSetArg> {
  using type = tuple_view<pi_kernel, pi_uint32, size_t, const void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetInfo> {
  using type = tuple_view<pi_kernel, pi_kernel_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetGroupInfo> {
  using type = tuple_view<pi_kernel, pi_device, pi_kernel_group_info, size_t,
                          void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelGetSubGroupInfo> {
  using type = tuple_view<pi_kernel, pi_device, pi_kernel_sub_group_info,
                          size_t, const void *, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelRetain> {
  using type = tuple_view<pi_kernel>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelRelease> {
  using type = tuple_view<pi_kernel>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgPointer> {
  using type = tuple_view<pi_kernel, pi_uint32, size_t, const void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piKernelSetExecInfo> {
  using type = tuple_view<pi_kernel, pi_kernel_exec_info, size_t, const void *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextKernelCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_context, bool, pi_kernel *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextKernelGetNativeHandle> {
  using type = tuple_view<pi_kernel, pi_native_handle *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventCreate> {
  using type = tuple_view<pi_context, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventGetInfo> {
  using type = tuple_view<pi_event, pi_event_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventGetProfilingInfo> {
  using type =
      tuple_view<pi_event, pi_profiling_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventsWait> {
  using type = tuple_view<pi_uint32, const pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventSetCallback> {
  using type = tuple_view<pi_event, pi_int32,
                          void (*)(pi_event, pi_int32, void *), void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventSetStatus> {
  using type = tuple_view<pi_event, pi_int32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventRetain> {
  using type = tuple_view<pi_event>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEventRelease> {
  using type = tuple_view<pi_event>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextEventGetNativeHandle> {
  using type = tuple_view<pi_event, pi_native_handle *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piextEventCreateWithNativeHandle> {
  using type = tuple_view<pi_native_handle, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerCreate> {
  using type =
      tuple_view<pi_context, const pi_sampler_properties *, pi_sampler *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerGetInfo> {
  using type =
      tuple_view<pi_sampler, pi_sampler_info, size_t, void *, size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerRetain> {
  using type = tuple_view<pi_sampler>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piSamplerRelease> {
  using type = tuple_view<pi_sampler>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueKernelLaunch> {
  using type =
      tuple_view<pi_queue, pi_kernel, pi_uint32, const size_t *, const size_t *,
                 const size_t *, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueNativeKernel> {
  using type = tuple_view<pi_queue, void (*)(void *), void *, size_t, pi_uint32,
                          const pi_mem *, const void **, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueEventsWait> {
  using type = tuple_view<pi_queue, pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueEventsWaitWithBarrier> {
  using type = tuple_view<pi_queue, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferRead> {
  using type = tuple_view<pi_queue, pi_mem, pi_bool, size_t, size_t, void *,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferReadRect> {
  using type = tuple_view<pi_queue, pi_mem, pi_bool, pi_buff_rect_offset,
                          pi_buff_rect_offset, pi_buff_rect_region, size_t,
                          size_t, size_t, size_t, void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferWrite> {
  using type =
      tuple_view<pi_queue, pi_mem, pi_bool, size_t, size_t, const void *,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferWriteRect> {
  using type = tuple_view<pi_queue, pi_mem, pi_bool, pi_buff_rect_offset,
                          pi_buff_rect_offset, pi_buff_rect_region, size_t,
                          size_t, size_t, size_t, const void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferCopy> {
  using type = tuple_view<pi_queue, pi_mem, pi_mem, size_t, size_t, size_t,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <>
struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferCopyRect> {
  using type =
      tuple_view<pi_queue, pi_mem, pi_mem, pi_buff_rect_offset,
                 pi_buff_rect_offset, pi_buff_rect_region, size_t, size_t,
                 size_t, size_t, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferFill> {
  using type =
      tuple_view<pi_queue, pi_mem, pi_bool, const void *, size_t, size_t,
                 size_t, pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageRead> {
  using type = tuple_view<pi_queue, pi_mem, pi_bool, pi_image_offset,
                          pi_image_region, size_t, size_t, void *, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageWrite> {
  using type = tuple_view<pi_queue, pi_mem, pi_bool, pi_image_offset,
                          pi_image_region, size_t, size_t, const void *,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageCopy> {
  using type =
      tuple_view<pi_queue, pi_mem, pi_mem, pi_image_offset, pi_image_offset,
                 pi_image_region, pi_uint32, const pi_event, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemImageFill> {
  using type =
      tuple_view<pi_queue, pi_mem, const void *, const size_t *, const size_t *,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemBufferMap> {
  using type =
      tuple_view<pi_queue, pi_mem, pi_bool, pi_map_flags, size_t, size_t,
                 pi_uint32, const pi_event *, pi_event *, void **>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piEnqueueMemUnmap> {
  using type = tuple_view<pi_queue, pi_mem, void *, pi_uint32, const pi_event *,
                          pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgMemObj> {
  using type = tuple_view<pi_kernel, pi_uint32, const pi_mem *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextKernelSetArgSampler> {
  using type = tuple_view<pi_kernel, pi_uint32, const pi_sampler *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMHostAlloc> {
  using type = tuple_view<void **, pi_context, pi_usm_mem_properties *, size_t,
                          pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMDeviceAlloc> {
  using type = tuple_view<void **, pi_context, pi_device,
                          pi_usm_mem_properties *, size_t, pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMSharedAlloc> {
  using type = tuple_view<void **, pi_context, pi_device,
                          pi_usm_mem_properties *, size_t, pi_uint32>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMFree> {
  using type = tuple_view<pi_context, void *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemset> {
  using type = tuple_view<pi_queue, void *, pi_int32, size_t, pi_uint32,
                          const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemcpy> {
  using type = tuple_view<pi_queue, pi_bool, void *, const void *, size_t,
                          pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueuePrefetch> {
  using type =
      tuple_view<pi_queue, const void *, size_t, pi_usm_migration_flags,
                 pi_uint32, const pi_event *, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMEnqueueMemAdvise> {
  using type =
      tuple_view<pi_queue, const void *, size_t, pi_mem_advice, pi_event *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextUSMGetMemAllocInfo> {
  using type = tuple_view<pi_context, const void *, pi_mem_info, size_t, void *,
                          size_t *>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piextPluginGetOpaqueData> {
  using type = tuple_view<void *, void **>;
};

template <> struct PiApiArgTuple<detail::PiApiKind::piTearDown> {
  using type = tuple_view<void *>;
};
} // namespace xpti_helpers
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
