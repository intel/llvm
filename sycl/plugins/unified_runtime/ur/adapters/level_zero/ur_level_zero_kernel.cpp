//===--------- ur_level_zero_kernel.cpp - Level Zero Adapter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_kernel.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t workDim, ///< [in] number of dimensions, from 1 to 3, to specify
                      ///< the global and work-group work-items
    const size_t
        *GlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned
                           ///< values that specify the offset used to
                           ///< calculate the global ID of a work-item
    const size_t *GlobalWorkSize, ///< [in] pointer to an array of workDim
                                  ///< unsigned values that specify the number
                                  ///< of global work-items in workDim that
                                  ///< will execute the kernel function
    const size_t
        *LocalWorkSize, ///< [in][optional] pointer to an array of workDim
                        ///< unsigned values that specify the number of local
                        ///< work-items forming a work-group that will execute
                        ///< the kernel function. If nullptr, the runtime
                        ///< implementation will choose the work-group size.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t Queue,     ///< [in] handle of the queue to submit to.
    ur_program_handle_t Program, ///< [in] handle of the program containing the
                                 ///< device global variable.
    const char
        *Name, ///< [in] the unique identifier for the device global variable.
    bool BlockingWrite, ///< [in] indicates if this operation should block.
    size_t Count,       ///< [in] the number of bytes to copy.
    size_t Offset, ///< [in] the byte offset into the device global variable to
                   ///< start copying.
    const void *Src, ///< [in] pointer to where the data must be copied from.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t Queue,     ///< [in] handle of the queue to submit to.
    ur_program_handle_t Program, ///< [in] handle of the program containing the
                                 ///< device global variable.
    const char
        *Name, ///< [in] the unique identifier for the device global variable.
    bool BlockingRead, ///< [in] indicates if this operation should block.
    size_t Count,      ///< [in] the number of bytes to copy.
    size_t Offset, ///< [in] the byte offset into the device global variable to
                   ///< start copying.
    void *Dst,     ///< [in] pointer to where the data must be copied to.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t Program, ///< [in] handle of the program instance
    const char *KernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *Kernel ///< [out] pointer to handle of kernel object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,    ///< [in] size of argument type
    const void
        *ArgValue ///< [in] argument value represented as matching arg type.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize     ///< [in] size of the local buffer to be allocated by the
                       ///< runtime
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(
    ur_kernel_handle_t Kernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t PropName, ///< [in] name of the Kernel property to query
    size_t PropSize,           ///< [in] the size of the Kernel property value.
    void *KernelInfo, ///< [in,out][optional] array of bytes holding the kernel
                      ///< info property. If propSize is not equal to or
                      ///< greater than the real number of bytes needed to
                      ///< return the info then the
                      ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                      ///< pKernelInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetGroupInfo(
    ur_kernel_handle_t Kernel, ///< [in] handle of the Kernel object
    ur_device_handle_t Device, ///< [in] handle of the Device object
    ur_kernel_group_info_t
        PropName,       ///< [in] name of the work Group property to query
    size_t PropSize,    ///< [in] size of the Kernel Work Group property value
    void *PropValue,    ///< [in,out][optional][range(0, propSize)] value of the
                        ///< Kernel Work Group property.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSubGroupInfo(
    ur_kernel_handle_t Kernel, ///< [in] handle of the Kernel object
    ur_device_handle_t Device, ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t
        PropName,       ///< [in] name of the SubGroup property to query
    size_t PropSize,    ///< [in] size of the Kernel SubGroup property value
    void *PropValue,    ///< [in,out][range(0, propSize)][optional] value of the
                        ///< Kernel SubGroup property.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to retain
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to release
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex,   ///< [in] argument index in range [0, num args - 1]
    const void *ArgValue ///< [in][optional] SVM pointer to memory location
                         ///< holding the argument value. If null then argument
                         ///< value is considered null.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex,   ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,      ///< [in] size of argument type
    const void *ArgValue ///< [in][optional] SVM pointer to memory location
                         ///< holding the argument value. If null then argument
                         ///< value is considered null.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t Kernel,      ///< [in] handle of the kernel object
    ur_kernel_exec_info_t PropName, ///< [in] name of the execution attribute
    size_t PropSize,                ///< [in] size in byte the attribute value
    const void *PropValue ///< [in][range(0, propSize)] pointer to memory
                          ///< location holding the property value.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    ur_sampler_handle_t ArgValue ///< [in] handle of Sampler object.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex,       ///< [in] argument index in range [0, num args - 1]
    ur_mem_handle_t ArgValue ///< [in][optional] handle of Memory object.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *NativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t NativeKernel, ///< [in] the native handle of the kernel.
    ur_context_handle_t Context,     ///< [in] handle of the context object
    ur_kernel_handle_t
        *Kernel ///< [out] pointer to the handle of the kernel object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t Count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t
        *SpecConstants ///< [in] array of specialization constant value
                       ///< descriptions
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}