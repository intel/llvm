#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/detail/cg_types.hpp> // NDRDescT
#include <sycl/detail/native_cpu.hpp>
#include <sycl/detail/pi.h>

#include "pi_native_cpu.hpp"

static bool PrintPiTrace = true;

struct _pi_object {
  _pi_object() : RefCount{1} {}

  std::atomic<pi_uint32> RefCount;
};

struct _pi_program : _pi_object {
  _pi_context *_ctx;
  const unsigned char *_ptr;
};

using nativecpu_kernel_t = void(const sycl::detail::NativeCPUArgDesc *,
                                __nativecpu_state *);
using nativecpu_ptr_t = nativecpu_kernel_t *;
using nativecpu_task_t = std::function<nativecpu_kernel_t>;
struct _pi_kernel : _pi_object {
  const char *_name;
  nativecpu_task_t _subhandler;
  std::vector<sycl::detail::NativeCPUArgDesc> _args;
};

// taken from pi_cuda.cpp
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {
    if (param_value_size < value_size) {
      return PI_ERROR_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    // Ignore unused parameter
    (void)value_size;

    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

sycl::detail::NDRDescT getNDRDesc(pi_uint32 WorkDim,
                                  const size_t *GlobalWorkOffset,
                                  const size_t *GlobalWorkSize,
                                  const size_t *LocalWorkSize) {
  // Todo: we flip indexes here, I'm not sure we should, if we don't we need to
  // un-flip them in the spirv builtins definitions as well
  sycl::detail::NDRDescT Res;
  switch (WorkDim) {
  case 1:
    Res.set<1>(sycl::nd_range<1>({GlobalWorkSize[0]}, {LocalWorkSize[0]},
                                 {GlobalWorkOffset[0]}));
    break;
  case 2:
    Res.set<2>(sycl::nd_range<2>({GlobalWorkSize[0], GlobalWorkSize[1]},
                                 {LocalWorkSize[0], LocalWorkSize[1]},
                                 {GlobalWorkOffset[0], GlobalWorkOffset[1]}));
    break;
  case 3:
    Res.set<3>(sycl::nd_range<3>(
        {GlobalWorkSize[0], GlobalWorkSize[1], GlobalWorkSize[2]},
        {LocalWorkSize[0], LocalWorkSize[1], LocalWorkSize[2]},
        {GlobalWorkOffset[0], GlobalWorkOffset[1], GlobalWorkOffset[2]}));
    break;
  }
  return Res;
}

extern "C" {
#define DIE_NO_IMPLEMENTATION                                                  \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Not Implemented : " << __FUNCTION__                          \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_ERROR_INVALID_OPERATION;

#define CONTINUE_NO_IMPLEMENTATION                                             \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Warning : Not Implemented : " << __FUNCTION__                \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_SUCCESS;

#define CASE_PI_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    if (PrintPiTrace) {                                                        \
      std::cerr << std::endl                                                   \
                << "Unsupported PI case : " << #not_supported << " in "        \
                << __FUNCTION__ << ":" << __LINE__ << "(" << __FILE__ << ")"   \
                << std::endl;                                                  \
    }                                                                          \
    return PI_ERROR_INVALID_OPERATION;

pi_result piextPlatformGetNativeHandle(pi_platform, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle, pi_platform *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCreateWithBinary(pi_context context, pi_uint32,
                                    const pi_device *, const size_t *,
                                    const unsigned char **binaries, size_t,
                                    const pi_device_binary_property *,
                                    pi_int32 *, pi_program *program) {
  // Todo: proper error checking
  assert(binaries);
  auto p = new _pi_program();
  p->_ptr = binaries[0];
  p->_ctx = context;
  *program = p;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithBinary(pi_context, pi_uint32, const pi_device *,
                                      const size_t *, const unsigned char **,
                                      pi_int32 *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piclProgramCreateWithSource(pi_context, pi_uint32, const char **,
                                      const size_t *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramGetInfo(pi_program program, pi_program_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret) {
  assert(program != nullptr);

  switch (param_name) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   &program->RefCount);
  case PI_PROGRAM_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_PROGRAM_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  case PI_PROGRAM_INFO_DEVICES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        program->_ctx->Device);
  case PI_PROGRAM_INFO_SOURCE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  case PI_PROGRAM_INFO_BINARY_SIZES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        "foo");
  case PI_PROGRAM_INFO_BINARIES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        "foo");
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "foo");
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result piProgramLink(pi_context, pi_uint32, const pi_device *, const char *,
                        pi_uint32, const pi_program *,
                        void (*)(pi_program, void *), void *, pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramCompile(pi_program, pi_uint32, const pi_device *,
                           const char *, pi_uint32, const pi_program *,
                           const char **, void (*)(pi_program, void *),
                           void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piProgramBuild(pi_program, pi_uint32, const pi_device *, const char *,
                         void (*)(pi_program, void *), void *) {
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program, pi_device,
                                pi_program_build_info param_name, size_t,
                                void *, size_t *) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piProgramRetain(pi_program) { DIE_NO_IMPLEMENTATION; }

pi_result piProgramRelease(pi_program program) {
  delete program;
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                             pi_program *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelCreate(pi_program program, const char *name,
                         pi_kernel *kernel) {
  // Todo: error checking
  auto ker = new _pi_kernel();
  auto f = reinterpret_cast<nativecpu_ptr_t>(program->_ptr);
  ker->_subhandler = *f;
  ker->_name = name;
  *kernel = ker;
  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel kernel, pi_uint32, size_t, const void *arg) {
  // Todo: error checking
  // Todo: I think that the opencl spec (and therefore the pi spec mandates that
  // arg is copied (this is why it is defined as const void*, I guess we should
  // do it
  kernel->_args.emplace_back(const_cast<void *>(arg));
  return PI_SUCCESS;
}

pi_result piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32,
                                  const pi_mem_obj_property *,
                                  const pi_mem *memObj) {
  // Todo: error checking
  _pi_mem *memPtr = *memObj;
  kernel->_args.emplace_back(memPtr->_mem);
  return PI_SUCCESS;
}

pi_result piextKernelSetArgSampler(pi_kernel, pi_uint32, const pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetInfo(pi_kernel, pi_kernel_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelGetGroupInfo(pi_kernel kernel, pi_device,
                               pi_kernel_group_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  // Todo: return something meaningful here, we could emit info in our
  // integration header and read them here?

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
      size_t global_work_size[3] = {0, 0, 0};

      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, global_work_size);
    }
    case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
      size_t max_threads = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     size_t(max_threads));
    }
    case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
      size_t group_size[3] = {1, 1, 1};
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, group_size);
    }
    case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
      int bytes = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      int warpSize = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<size_t>(warpSize));
    }
    case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
      int bytes = 0;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_NUM_REGS: {
      sycl::detail::pi::die("PI_KERNEL_GROUP_INFO_NUM_REGS in "
                            "piKernelGetGroupInfo not implemented\n");
      return {};
    }

    default:
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_ERROR_INVALID_KERNEL;
}

pi_result piKernelGetSubGroupInfo(pi_kernel, pi_device,
                                  pi_kernel_sub_group_info, size_t,
                                  const void *, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelRetain(pi_kernel) { DIE_NO_IMPLEMENTATION; }

pi_result piKernelRelease(pi_kernel kernel) {
  delete kernel;
  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context, pi_event *) { DIE_NO_IMPLEMENTATION; }

pi_result piEventGetInfo(pi_event, pi_event_info, size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  // Todo: currently we do everything synchronously so this is a no-op
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event, pi_int32,
                             void (*)(pi_event, pi_int32, void *), void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEventSetStatus(pi_event, pi_int32) { DIE_NO_IMPLEMENTATION; }

pi_result piEventRetain(pi_event Event) { DIE_NO_IMPLEMENTATION; }

pi_result piEventRelease(pi_event Event) { DIE_NO_IMPLEMENTATION; }

pi_result piextEventGetNativeHandle(pi_event, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                           pi_event *) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piSamplerCreate(pi_context, const pi_sampler_properties *,
                          pi_sampler *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerGetInfo(pi_sampler, pi_sampler_info, size_t, void *,
                           size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piSamplerRetain(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piSamplerRelease(pi_sampler) { DIE_NO_IMPLEMENTATION; }

pi_result piEnqueueEventsWait(pi_queue, pi_uint32, const pi_event *,
                              pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32, const pi_event *,
                                         pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  // TODO: is it ok to have this as no-op?
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferReadRect(pi_queue, pi_mem, pi_bool,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, void *, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWrite(pi_queue, pi_mem, pi_bool, size_t, size_t,
                                  const void *, pi_uint32, const pi_event *,
                                  pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferWriteRect(pi_queue, pi_mem, pi_bool,
                                      pi_buff_rect_offset, pi_buff_rect_offset,
                                      pi_buff_rect_region, size_t, size_t,
                                      size_t, size_t, const void *, pi_uint32,
                                      const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopy(pi_queue, pi_mem, pi_mem, size_t, size_t,
                                 size_t, pi_uint32, const pi_event *,
                                 pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferCopyRect(pi_queue, pi_mem, pi_mem,
                                     pi_buff_rect_offset, pi_buff_rect_offset,
                                     pi_buff_rect_region, size_t, size_t,
                                     size_t, size_t, pi_uint32,
                                     const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemBufferFill(pi_queue, pi_mem buffer, const void *pattern,
                                 size_t pattern_size, size_t offset,
                                 size_t size, pi_uint32, const pi_event *,
                                 pi_event *) {
  // Todo: error checking
  // Todo: handle async
  void *startingPtr = buffer->_mem + offset;
  unsigned steps = size / pattern_size;
  for (unsigned i = 0; i < steps; i++) {
    memcpy(static_cast<int8_t *>(startingPtr) + i * pattern_size, pattern,
           pattern_size);
  }
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferMap(pi_queue, pi_mem buffer, pi_bool, pi_map_flags,
                                size_t offset, size_t size, pi_uint32,
                                const pi_event *, pi_event *, void **ret_map) {
  // Todo: add proper error checking
  *ret_map = buffer->_mem + offset;
  return PI_SUCCESS;
}

pi_result piEnqueueMemUnmap(pi_queue, pi_mem mem_obj, void *mapped_ptr,
                            pi_uint32, const pi_event *, pi_event *) {
  // Todo: no-op?
  return PI_SUCCESS;
}

pi_result piEnqueueMemImageRead(pi_queue CommandQueue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageWrite(pi_queue, pi_mem, pi_bool, pi_image_offset,
                                 pi_image_region, size_t, size_t, const void *,
                                 pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageCopy(pi_queue, pi_mem, pi_mem, pi_image_offset,
                                pi_image_offset, pi_image_region, pi_uint32,
                                const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueMemImageFill(pi_queue, pi_mem, const void *, const size_t *,
                                const size_t *, pi_uint32, const pi_event *,
                                pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  // TODO: add proper error checking
  // TODO: add proper event dep management
  sycl::detail::NDRDescT ndr =
      getNDRDesc(WorkDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize);
  __nativecpu_state state(ndr.GlobalSize[0], ndr.GlobalSize[1],
                          ndr.GlobalSize[2], ndr.LocalSize[0], ndr.LocalSize[1],
                          ndr.LocalSize[2], ndr.GlobalOffset[0],
                          ndr.GlobalOffset[1], ndr.GlobalOffset[2]);
  auto numWG0 = ndr.GlobalSize[0] / ndr.LocalSize[0];
  auto numWG1 = ndr.GlobalSize[1] / ndr.LocalSize[1];
  auto numWG2 = ndr.GlobalSize[2] / ndr.LocalSize[2];
  for (unsigned g2 = 0; g2 < numWG2; g2++) {
    for (unsigned g1 = 0; g1 < numWG1; g1++) {
      for (unsigned g0 = 0; g0 < numWG0; g0++) {
        for (unsigned local2 = 0; local2 < ndr.LocalSize[2]; local2++) {
          for (unsigned local1 = 0; local1 < ndr.LocalSize[1]; local1++) {
            for (unsigned local0 = 0; local0 < ndr.LocalSize[0]; local0++) {
              state.update(g0, g1, g2, local0, local1, local2);
              Kernel->_subhandler(Kernel->_args.data(), &state);
            }
          }
        }
      }
    }
  }
  // Todo: we should avoid calling clear here by avoiding using push_back
  // in setKernelArgs.
  Kernel->_args.clear();
  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle, pi_context,
                                            pi_program, bool, pi_kernel *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextKernelGetNativeHandle(pi_kernel, pi_native_handle *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                pi_uint32, const pi_mem *, const void **,
                                pi_uint32, const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextGetDeviceFunctionPointer(pi_device, pi_program, const char *,
                                        pi_uint64 *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMHostAlloc(void **result_ptr, pi_context,
                            pi_usm_mem_properties *, size_t size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *result_ptr = malloc(size);
  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context, pi_device,
                              pi_usm_mem_properties *, size_t Size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *ResultPtr = malloc(Size);
  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMFree(pi_context, void *Ptr) {
  // Todo: error checking
  free(Ptr);
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel kernel, pi_uint32, size_t,
                                   const void *ptr) {
  // Todo: error checking
  auto PtrToPtr = reinterpret_cast<const intptr_t *>(ptr);
  auto DerefPtr = reinterpret_cast<void *>(*PtrToPtr);
  kernel->_args.push_back(DerefPtr);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemset(pi_queue, void *ptr, pi_int32 value,
                                size_t count, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking.
  memset(ptr, value, count);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *dest, const void *src,
                                size_t len, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking
  memcpy(dest, src, len);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue, const void *, size_t,
                                   pi_mem_advice, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMGetMemAllocInfo(pi_context, const void *, pi_mem_alloc_info,
                                  size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                              const void *) {
  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program, pi_uint32, size_t,
                                                const void *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueuePrefetch(pi_queue, const void *, size_t,
                                  pi_usm_migration_flags, pi_uint32,
                                  const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPluginGetOpaqueData(void *, void **OpaqueDataReturn) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreate(pi_context context, pi_device device,
                           pi_queue_properties *properties, pi_queue *queue) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueWriteHostPipe(pi_queue queue, pi_program program,
                                    const char *pipe_symbol, pi_bool blocking,
                                    void *ptr, size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueReadHostPipe(pi_queue queue, pi_program program,
                                   const char *pipe_symbol, pi_bool blocking,
                                   void *ptr, size_t size,
                                   pi_uint32 num_events_in_waitlist,
                                   const pi_event *events_waitlist,
                                   pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piPluginGetLastError(char **message) { DIE_NO_IMPLEMENTATION; }

pi_result piextUSMEnqueueMemcpy2D(pi_queue queue, pi_bool blocking,
                                  void *dst_ptr, size_t dst_pitch,
                                  const void *src_ptr, size_t src_pitch,
                                  size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue queue, pi_program program, const char *name,
    pi_bool blocking_write, size_t count, size_t offset, const void *src,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue queue, pi_program program, const char *name, pi_bool blocking_read,
    size_t count, size_t offset, void *dst, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piPluginGetBackendOption(pi_platform platform,
                                   const char *frontend_option,
                                   const char **backend_option) {
  CONTINUE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemset2D(pi_queue queue, void *ptr, size_t pitch,
                                  int value, size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                  uint64_t *HostTime) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueGetNativeHandle2(pi_queue queue,
                                     pi_native_handle *nativeHandle,
                                     int32_t *nativeHandleDesc) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextQueueCreate2(pi_context context, pi_device device,
                            pi_queue_properties *properties, pi_queue *queue) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle2(
    pi_native_handle nativeHandle, int32_t nativeHandleDesc, pi_context context,
    pi_device device, bool pluginOwnsNativeHandle,
    pi_queue_properties *Properties, pi_queue *queue) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueFill2D(pi_queue queue, void *ptr, size_t pitch,
                                size_t pattern_size, const void *pattern,
                                size_t width, size_t height,
                                pi_uint32 num_events_in_waitlist,
                                const pi_event *events_waitlist,
                                pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piextCommandBufferCreate(pi_context, pi_device,
                                   const pi_ext_command_buffer_desc *,
                                   pi_ext_command_buffer *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRetain(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRelease(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferFinalize(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferNDRangeKernel(pi_ext_command_buffer, pi_kernel,
                                          pi_uint32, const size_t *,
                                          const size_t *, const size_t *,
                                          pi_uint32, const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemcpyUSM(pi_ext_command_buffer, void *,
                                      const void *, size_t, pi_uint32,
                                      const pi_ext_sync_point *,
                                      pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopy(pi_ext_command_buffer, pi_mem, pi_mem,
                                          size_t, size_t, size_t, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer, pi_mem, pi_mem, pi_buff_rect_offset,
    pi_buff_rect_offset, pi_buff_rect_region, size_t, size_t, size_t, size_t,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferRead(pi_ext_command_buffer, pi_mem, size_t,
                                          size_t, void *, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, void *, pi_uint32,
    const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWrite(pi_ext_command_buffer, pi_mem,
                                           size_t, size_t, const void *,
                                           pi_uint32, const pi_ext_sync_point *,
                                           pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, const void *,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer, pi_queue, pi_uint32,
                                    const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnablePeerAccess(pi_device, pi_device) { DIE_NO_IMPLEMENTATION; }

pi_result piextDisablePeerAccess(pi_device, pi_device) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPeerAccessGetInfo(pi_device, pi_device, pi_peer_attr, size_t,
                                 void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piTearDown(void *) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {

#define _PI_CL(pi_api, native_cpu_api)                                         \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&native_cpu_api);

  // Platform
  _PI_CL(piPlatformsGet, pi2ur::piPlatformsGet)
  _PI_CL(piPlatformGetInfo, pi2ur::piPlatformGetInfo)
  // Device
  _PI_CL(piDevicesGet, pi2ur::piDevicesGet)
  _PI_CL(piDeviceGetInfo, pi2ur::piDeviceGetInfo)
  _PI_CL(piDevicePartition, pi2ur::piDevicePartition)
  _PI_CL(piDeviceRetain, pi2ur::piDeviceRetain)
  _PI_CL(piDeviceRelease, pi2ur::piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, pi2ur::piextDeviceSelectBinary)
  // TODO: uncomment when program is fully ported
  //  _PI_CL(piextGetDeviceFunctionPointer,
  //  pi2ur::piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, pi2ur::piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle,
         pi2ur::piextDeviceCreateWithNativeHandle)
  // Context
  _PI_CL(piextContextSetExtendedDeleter, pi2ur::piextContextSetExtendedDeleter)
  _PI_CL(piContextCreate, pi2ur::piContextCreate)
  _PI_CL(piContextGetInfo, pi2ur::piContextGetInfo)
  _PI_CL(piContextRetain, pi2ur::piContextRetain)
  _PI_CL(piContextRelease, pi2ur::piContextRelease)
  _PI_CL(piextContextGetNativeHandle, pi2ur::piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle,
         pi2ur::piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, pi2ur::piQueueCreate)
  _PI_CL(piextQueueCreate, pi2ur::piextQueueCreate)
  _PI_CL(piQueueGetInfo, pi2ur::piQueueGetInfo)
  _PI_CL(piQueueFinish, pi2ur::piQueueFinish)
  _PI_CL(piQueueFlush, pi2ur::piQueueFlush)
  _PI_CL(piQueueRetain, pi2ur::piQueueRetain)
  _PI_CL(piQueueRelease, pi2ur::piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, pi2ur::piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle,
         pi2ur::piextQueueCreateWithNativeHandle)

  // Memory
  _PI_CL(piMemBufferCreate, pi2ur::piMemBufferCreate)
  _PI_CL(piMemImageCreate, pi2ur::piMemImageCreate)
  _PI_CL(piMemGetInfo, pi2ur::piMemGetInfo)
  _PI_CL(piMemImageGetInfo, pi2ur::piMemImageGetInfo)
  _PI_CL(piMemRetain, pi2ur::piMemRetain)
  _PI_CL(piMemRelease, pi2ur::piMemRelease)
  _PI_CL(piMemBufferPartition, pi2ur::piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, pi2ur::piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, pi2ur::piextMemCreateWithNativeHandle)

  // Program
  _PI_CL(piProgramCreate, piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, piProgramGetInfo)
  _PI_CL(piProgramCompile, piProgramCompile)
  _PI_CL(piProgramBuild, piProgramBuild)
  _PI_CL(piProgramLink, piProgramLink)
  _PI_CL(piProgramGetBuildInfo, piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, piProgramRetain)
  _PI_CL(piProgramRelease, piProgramRelease)
  _PI_CL(piextProgramGetNativeHandle, piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle, piextProgramCreateWithNativeHandle)
  _PI_CL(piextProgramSetSpecializationConstant,
         piextProgramSetSpecializationConstant)
  // Kernel
  _PI_CL(piKernelCreate, piKernelCreate)
  _PI_CL(piKernelSetArg, piKernelSetArg)
  _PI_CL(piKernelGetInfo, piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, piKernelRetain)
  _PI_CL(piKernelRelease, piKernelRelease)
  _PI_CL(piextKernelGetNativeHandle, piextKernelGetNativeHandle)
  _PI_CL(piKernelSetExecInfo, piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, piextKernelSetArgPointer)
  _PI_CL(piextKernelCreateWithNativeHandle, piextKernelCreateWithNativeHandle)

  // Event
  _PI_CL(piEventCreate, piEventCreate)
  _PI_CL(piEventGetInfo, piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, piEventGetProfilingInfo)
  _PI_CL(piEventsWait, piEventsWait)
  _PI_CL(piEventSetCallback, piEventSetCallback)
  _PI_CL(piEventSetStatus, piEventSetStatus)
  _PI_CL(piEventRetain, piEventRetain)
  _PI_CL(piEventRelease, piEventRelease)
  _PI_CL(piextEventGetNativeHandle, piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle, piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, piSamplerCreate)
  _PI_CL(piSamplerGetInfo, piSamplerGetInfo)
  _PI_CL(piSamplerRetain, piSamplerRetain)
  _PI_CL(piSamplerRelease, piSamplerRelease)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, piEnqueueEventsWait)
  _PI_CL(piEnqueueEventsWaitWithBarrier, piEnqueueEventsWaitWithBarrier)
  _PI_CL(piEnqueueMemBufferRead, piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, piEnqueueMemUnmap)

  // USM
  _PI_CL(piextUSMHostAlloc, piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueFill2D, piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMEnqueueMemcpy2D, piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMGetMemAllocInfo, piextUSMGetMemAllocInfo)
  // Device global variable
  _PI_CL(piextEnqueueDeviceGlobalVariableWrite,
         piextEnqueueDeviceGlobalVariableWrite)
  _PI_CL(piextEnqueueDeviceGlobalVariableRead,
         piextEnqueueDeviceGlobalVariableRead)

  // Host Pipe
  _PI_CL(piextEnqueueReadHostPipe, piextEnqueueReadHostPipe)
  _PI_CL(piextEnqueueWriteHostPipe, piextEnqueueWriteHostPipe)

  _PI_CL(piextKernelSetArgMemObj, piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, piextKernelSetArgSampler)
  _PI_CL(piPluginGetLastError, piPluginGetLastError)
  _PI_CL(piTearDown, piTearDown)
  _PI_CL(piGetDeviceAndHostTimer, piGetDeviceAndHostTimer)
  _PI_CL(piPluginGetBackendOption, piPluginGetBackendOption)

#undef _PI_CL
  return PI_SUCCESS;
}
}
