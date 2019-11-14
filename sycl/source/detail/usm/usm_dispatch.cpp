//==------------ usm_dispatch.cpp - USM Dispatch Impl ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

/***

  General philosophy: Try to use a CL extension for each function,
   if it exists. Otherwise, fall back to CLUSM's USM-on-SVM.

 **/
#define GET_EXTENSION(_funcname)                                               \
  pfn_##_funcname = (_funcname##_fn)clGetExtensionFunctionAddressForPlatform(  \
      platform, #_funcname);

USMDispatcher::USMDispatcher(cl_platform_id platform,
                             const vector_class<RT::PiDevice> &DeviceIds) {
  // Note: This function should be modified whenever a new BE is added.
  // mSupported needs to be appropriately set to properly gate USM support.
  
  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    GET_EXTENSION(clHostMemAllocINTEL);
    GET_EXTENSION(clDeviceMemAllocINTEL);
    GET_EXTENSION(clSharedMemAllocINTEL);
    GET_EXTENSION(clMemFreeINTEL);
    GET_EXTENSION(clGetMemAllocInfoINTEL);
    GET_EXTENSION(clSetKernelArgMemPointerINTEL);
    GET_EXTENSION(clEnqueueMemsetINTEL);
    GET_EXTENSION(clEnqueueMemcpyINTEL);
    GET_EXTENSION(clEnqueueMigrateMemINTEL);
    GET_EXTENSION(clEnqueueMemAdviseINTEL);
    mEmulated = !(pfn_clHostMemAllocINTEL && pfn_clDeviceMemAllocINTEL &&
                  pfn_clSharedMemAllocINTEL && pfn_clMemFreeINTEL &&
                  pfn_clSetKernelArgMemPointerINTEL &&
                  pfn_clEnqueueMemsetINTEL && pfn_clEnqueueMemcpyINTEL);
    mEmulator.reset(new CLUSM());

    if (mEmulated) {
      // See if every device in this context supports
      // CL_DEVICE_SVM_FINE_GRAIN_BUFFER
      // If not, disable USM

      if (CL_TARGET_OPENCL_VERSION >= 200) {
        bool AnybodyNotSupportSVM = false;
        for (const auto &D : DeviceIds) {
          cl_device_svm_capabilities Caps;
          cl_int Error = clGetDeviceInfo(
              pi::cast<cl_device_id>(D), CL_DEVICE_SVM_CAPABILITIES,
              sizeof(cl_device_svm_capabilities), &Caps, nullptr);
          AnybodyNotSupportSVM |= ((Error != CL_SUCCESS) ||
                                   (!(Caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)));
        }
        mSupported = !AnybodyNotSupportSVM;
      } else {
        // USM isn't support on CL 1.2
        mSupported = false;
      }
    } else {
      // We support the CL Extension
      mSupported = true;
    }
  } else {
    mSupported = false;
  }
}

void *USMDispatcher::hostMemAlloc(pi_context Context,
                                  cl_mem_properties_intel *Properties,
                                  size_t Size, pi_uint32 Alignment,
                                  pi_result *ErrcodeRet) {
  void *RetVal = nullptr;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_context CLContext = pi::cast<cl_context>(Context);

    if (mEmulated) {
      RetVal = mEmulator->hostMemAlloc(CLContext, Properties, Size, Alignment,
                                       pi::cast<cl_int *>(ErrcodeRet));
    } else {
      RetVal = pfn_clHostMemAllocINTEL(CLContext, Properties, Size, Alignment,
                                       pi::cast<cl_int *>(ErrcodeRet));
    }
  }

  if (ErrcodeRet && !RetVal) {
    *ErrcodeRet = PI_INVALID_OPERATION;
  }
  return RetVal;
}

void *USMDispatcher::deviceMemAlloc(pi_context Context, pi_device Device,
                                    cl_mem_properties_intel *Properties,
                                    size_t Size, pi_uint32 Alignment,
                                    pi_result *ErrcodeRet) {
  void *RetVal = nullptr;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_context CLContext = pi::cast<cl_context>(Context);
    cl_device_id CLDevice = pi::cast<cl_device_id>(Device);

    if (mEmulated) {
      RetVal = mEmulator->deviceMemAlloc(CLContext, CLDevice, Properties, Size,
                                         Alignment,
                                         pi::cast<cl_int *>(ErrcodeRet));
    } else {
      RetVal = pfn_clDeviceMemAllocINTEL(CLContext, CLDevice, Properties, Size,
                                         Alignment,
                                         pi::cast<cl_int *>(ErrcodeRet));
    }
  }

  if (ErrcodeRet && !RetVal) {
    *ErrcodeRet = PI_INVALID_OPERATION;
  }
  return RetVal;
}

void *USMDispatcher::sharedMemAlloc(pi_context Context, pi_device Device,
                                    cl_mem_properties_intel *Properties,
                                    size_t Size, pi_uint32 Alignment,
                                    pi_result *ErrcodeRet) {
  void *RetVal = nullptr;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_context CLContext = pi::cast<cl_context>(Context);
    cl_device_id CLDevice = pi::cast<cl_device_id>(Device);

    if (mEmulated) {
      RetVal = mEmulator->sharedMemAlloc(CLContext, CLDevice, Properties, Size,
                                         Alignment,
                                         pi::cast<cl_int *>(ErrcodeRet));
    } else {
      RetVal = pfn_clSharedMemAllocINTEL(CLContext, CLDevice, Properties, Size,
                                         Alignment,
                                         pi::cast<cl_int *>(ErrcodeRet));
    }
  }

  if (ErrcodeRet && !RetVal) {
    *ErrcodeRet = PI_INVALID_OPERATION;
  }
  return RetVal;
}

pi_result USMDispatcher::memFree(pi_context Context, void *Ptr) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_context CLContext = pi::cast<cl_context>(Context);

    if (mEmulated) {
      RetVal = pi::cast<pi_result>(mEmulator->memFree(CLContext, Ptr));
    } else {
      RetVal = pi::cast<pi_result>(pfn_clMemFreeINTEL(CLContext, Ptr));
    }
  }

  return RetVal;
}

pi_result USMDispatcher::setKernelArgMemPointer(pi_kernel Kernel,
                                                pi_uint32 ArgIndex,
                                                const void *ArgValue) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (mSupported) {
    if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
      cl_kernel CLKernel = pi::cast<cl_kernel>(Kernel);

      if (mEmulated) {
        RetVal = pi::cast<pi_result>(
            clSetKernelArgSVMPointer(CLKernel, ArgIndex, ArgValue));
      } else {
        RetVal = pi::cast<pi_result>(
            pfn_clSetKernelArgMemPointerINTEL(CLKernel, ArgIndex, ArgValue));
      }
    }
  }

  return RetVal;
}

void USMDispatcher::setKernelIndirectAccess(pi_kernel Kernel, pi_queue Queue) {

  if (mSupported) {
    if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
      cl_kernel CLKernel = pi::cast<cl_kernel>(Kernel);
      cl_command_queue CLQueue = pi::cast<cl_command_queue>(Queue);
      cl_bool TrueVal = CL_TRUE;

      if (mEmulated) {
        CHECK_OCL_CODE(mEmulator->setKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
        CHECK_OCL_CODE(mEmulator->setKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
        CHECK_OCL_CODE(mEmulator->setKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
        CHECK_OCL_CODE(
            mEmulator->setKernelIndirectUSMExecInfo(CLQueue, CLKernel));
      } else {
        CHECK_OCL_CODE(clSetKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
        CHECK_OCL_CODE(clSetKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
        CHECK_OCL_CODE(clSetKernelExecInfo(
            CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
            sizeof(cl_bool), &TrueVal));
      }
    }
  }
}

pi_result USMDispatcher::enqueueMemset(pi_queue Queue, void *Ptr,
                                       pi_int32 Value, size_t Count,
                                       pi_uint32 NumEventsInWaitList,
                                       const pi_event *EventWaitList,
                                       pi_event *Event) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_command_queue CLQueue = pi::cast<cl_command_queue>(Queue);

    // Is there a better way to convert pi_event * to cl_event *?
    
    if (mEmulated) {
      const cl_uchar Pattern = (cl_uchar)Value;

      RetVal = pi::cast<pi_result>(clEnqueueSVMMemFill(
          CLQueue, Ptr, &Pattern, sizeof(Pattern), Count, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
    } else {
      RetVal = pi::cast<pi_result>(pfn_clEnqueueMemsetINTEL(
          CLQueue, Ptr, Value, Count, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
    }
  }

  return RetVal;
}

pi_result USMDispatcher::enqueueMemcpy(pi_queue Queue, pi_bool Blocking,
                                       void *DestPtr, const void *SrcPtr,
                                       size_t Size,
                                       pi_uint32 NumEventsInWaitList,
                                       const pi_event *EventWaitList,
                                       pi_event *Event) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_command_queue CLQueue = pi::cast<cl_command_queue>(Queue);

    if (mEmulated) {
      RetVal = pi::cast<pi_result>(clEnqueueSVMMemcpy(
          CLQueue, Blocking, DestPtr, SrcPtr, Size, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
    } else {
      RetVal = pi::cast<pi_result>(pfn_clEnqueueMemcpyINTEL(
          CLQueue, Blocking, DestPtr, SrcPtr, Size, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
    }
  }

  return RetVal;
}

pi_result USMDispatcher::enqueueMigrateMem(pi_queue Queue, const void *Ptr,
                                           size_t Size,
                                           cl_mem_migration_flags Flags,
                                           pi_uint32 NumEventsInWaitList,
                                           const pi_event *EventWaitList,
                                           pi_event *Event) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_command_queue CLQueue = pi::cast<cl_command_queue>(Queue);

    if (mEmulated) {
      // We could check for OpenCL 2.1 and call the SVM migrate
      // functions, but for now we'll just enqueue a marker.
      RetVal = pi::cast<pi_result>(clEnqueueMarkerWithWaitList(
          CLQueue, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
      pi::piCheckResult(RetVal);
    } else {
      RetVal = pi::cast<pi_result>(pfn_clEnqueueMigrateMemINTEL(
          CLQueue, Ptr, Size, Flags, NumEventsInWaitList,
          reinterpret_cast<const cl_event *>(EventWaitList),
          reinterpret_cast<cl_event *>(Event)));
    }
  }

  return RetVal;
}

pi_result USMDispatcher::getMemAllocInfo(pi_context Context, const void *Ptr,
                                         cl_mem_info_intel ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_context CLContext = pi::cast<cl_context>(Context);

    if (mEmulated) {
      // TODO: What should we do here?
      // This isn't really supported yet.
      // Advice is typically safe to ignore,
      //  so a NOP will do.
      RetVal = pi::cast<pi_result>(mEmulator->getMemAllocInfoINTEL(
          CLContext, Ptr, ParamName, ParamValueSize, ParamValue,
          ParamValueSizeRet));
    } else {
      RetVal = pi::cast<pi_result>(
          pfn_clGetMemAllocInfoINTEL(CLContext, Ptr, ParamName, ParamValueSize,
                                     ParamValue, ParamValueSizeRet));
    }
  }

  return RetVal;
}

void USMDispatcher::memAdvise(pi_queue Queue, const void *Ptr, size_t Length,
                              int Advice, pi_event *Event) {
  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    cl_command_queue CLQueue = pi::cast<cl_command_queue>(Queue);

    if (mEmulated) {
      // memAdvise does nothing here
      // TODO: Implement a PI call for this openCL API
      RT::piCheckResult(RT::cast<RT::PiResult>(clEnqueueMarkerWithWaitList(
          CLQueue, 0, nullptr, reinterpret_cast<cl_event *>(Event))));
    } else {
      // Temporary until driver supports
      // memAdvise doesn't do anything on an iGPU anyway
      // TODO: Implement a PI call for this openCL API
      RT::piCheckResult(RT::cast<RT::PiResult>(clEnqueueMarkerWithWaitList(
          CLQueue, 0, nullptr, reinterpret_cast<cl_event *>(Event))));
      /*
      // Enable once this is supported in the driver
      auto CLAdvice = *reinterpret_cast<cl_mem_advice_intel *>(&Advice);
      // TODO: Implement a PI call for this openCL API
      RT::piCheckResult(RT::cast<RT::PiResult>(pfn_clEnqueueMemAdviseINTEL,
          CLQueue, Ptr, Length, CLAdvice, 0, nullptr,
          reinterpret_cast<cl_event *>(Event)));
      */
    }
  }
}

pi_result USMDispatcher::enqueuePrefetch(pi_queue Queue, void *Ptr, size_t Size,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *Event) {
  pi_result RetVal = PI_INVALID_OPERATION;

  if (pi::useBackend(pi::Backend::SYCL_BE_PI_OPENCL)) {
    if (mEmulated) {
      // Prefetch is a hint, so ignoring it is always safe.
      RetVal = PI_CALL_RESULT(RT::piEnqueueEventsWait,
          Queue, NumEventsInWaitList, EventWaitList, Event);
    } else {
      // TODO: Replace this with real prefetch support when the driver enables
      // it.
      RetVal = PI_CALL_RESULT(RT::piEnqueueEventsWait,
          Queue, NumEventsInWaitList, EventWaitList, Event);
    }
  }

  return RetVal;
}

} // namespace usm
} // namespace detail
} // namespace sycl
} // namespace cl
