#include "kernel.hpp"
#include "program.hpp"
#include "ur2offload.hpp"
#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  ur_kernel_handle_t Kernel = new ur_kernel_handle_t_;

  auto Res = olGetKernel(hProgram->OffloadProgram, pKernelName,
                         &Kernel->OffloadKernel);

  if (Res != OL_SUCCESS) {
    delete Kernel;
    return offloadResultToUR(Res);
  }

  *phKernel = Kernel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  hKernel->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  if (--hKernel->RefCount == 0) {
    delete hKernel;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(ur_kernel_handle_t, ur_kernel_exec_info_t, size_t,
                    const ur_kernel_exec_info_properties_t *, const void *) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_pointer_properties_t *, const void *pArgValue) {
  hKernel->Args.addArg(argIndex, sizeof(pArgValue), &pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *, const void *pArgValue) {
  hKernel->Args.addArg(argIndex, argSize, pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t, ur_device_handle_t,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  if (propName == UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    size_t GroupSize[3] = {0, 0, 0};
    return ReturnValue(GroupSize, 3);
  }
  return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
}
