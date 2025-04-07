//===--------- memory_helpers.cpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mutable_helpers.hpp"


ur_result_t setMutableOffsetDesc(
    std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>> &Desc,
    uint32_t Dim, size_t *NewGlobalWorkOffset, const void *NextDesc,
    uint64_t CommandID) {
  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->offsetX = NewGlobalWorkOffset[0];
  Desc->offsetY = Dim >= 2 ? NewGlobalWorkOffset[1] : 0;
  Desc->offsetZ = Dim == 3 ? NewGlobalWorkOffset[2] : 0;
  return UR_RESULT_SUCCESS;
}

ur_result_t setMutableGroupSizeDesc(
    std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>> &Desc,
    uint32_t Dim, uint32_t *NewLocalWorkSize, const void *NextDesc,
    uint64_t CommandID) {
  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->groupSizeX = NewLocalWorkSize[0];
  Desc->groupSizeY = Dim >= 2 ? NewLocalWorkSize[1] : 1;
  Desc->groupSizeZ = Dim == 3 ? NewLocalWorkSize[2] : 1;
  return UR_RESULT_SUCCESS;
}

ur_result_t setMutableGroupCountDesc(
    std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>> &Desc,
    ze_group_count_t *ZeThreadGroupDimensions, const void *NextDesc,
    uint64_t CommandID) {
  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->pGroupCount = ZeThreadGroupDimensions;
  return UR_RESULT_SUCCESS;
}

ur_result_t setMutableMemObjArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    uint32_t argIndex, const void *pArgValue, const void *NextDesc,
    uint64_t CommandID) {

  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->argIndex = argIndex;
  Desc->argSize = sizeof(void *);
  Desc->pArgValue = pArgValue;
  return UR_RESULT_SUCCESS;
}

ur_result_t setMutablePointerArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    const ur_exp_command_buffer_update_pointer_arg_desc_t &NewPointerArgDesc,
    const void *NextDesc, uint64_t CommandID) {
  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->argIndex = NewPointerArgDesc.argIndex;
  Desc->argSize = sizeof(void *);
  Desc->pArgValue = NewPointerArgDesc.pNewPointerArg;
  return UR_RESULT_SUCCESS;
}

ur_result_t setMutableValueArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    const ur_exp_command_buffer_update_value_arg_desc_t &NewValueArgDesc,
    const void *NextDesc, uint64_t CommandID) {
  Desc->commandId = CommandID;
  Desc->pNext = NextDesc;
  Desc->argIndex = NewValueArgDesc.argIndex;
  Desc->argSize = NewValueArgDesc.argSize;
  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  const void *ArgValuePtr = NewValueArgDesc.pNewValueArg;
  if (NewValueArgDesc.argSize == sizeof(void *) && ArgValuePtr &&
      *(void **)(const_cast<void *>(ArgValuePtr)) == nullptr) {
    ArgValuePtr = nullptr;
  }
  Desc->pArgValue = ArgValuePtr;
  return UR_RESULT_SUCCESS;
}