//===--------- mutable_helpers.cpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mutable_helpers.hpp"
#include "../device.hpp"
#include "../ur_interface_loader.hpp"
#include "../ur_level_zero.hpp"
#include "kernel_helpers.hpp"

using desc_storage_t = std::vector<std::variant<
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>>>>;

namespace {

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

ur_result_t updateKernelSizes(
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    ze_group_count_t &zeThreadGroupDimensionsList,
    ur_result_t (*getZeKernel)(ur_kernel_handle_t, ze_kernel_handle_t &,
                               ur_device_handle_t),
    ur_device_handle_t device, desc_storage_t &descs) {
  uint32_t dim = commandDesc.newWorkDim;
  // Update global offset if provided.
  if (size_t *newGlobalWorkOffset = commandDesc.pNewGlobalWorkOffset;
      newGlobalWorkOffset && dim > 0) {
    auto mutableGroupOffestDesc =
        std::make_unique<ZeStruct<ze_mutable_global_offset_exp_desc_t>>();
    UR_CALL(setMutableOffsetDesc(mutableGroupOffestDesc, dim,
                                 newGlobalWorkOffset, *nextDesc,
                                 command->commandId));
    *nextDesc = mutableGroupOffestDesc.get();
    descs.push_back(std::move(mutableGroupOffestDesc));
  }

  // Update local-size/group-size if provided.
  size_t *newLocalWorkSize = commandDesc.pNewLocalWorkSize;
  if (newLocalWorkSize && dim > 0) {
    auto mutableGroupSizeDesc =
        std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();

    uint32_t workgroupSize[3] = {1, 1, 1};
    for (size_t d = 0; d < dim; d++) {
      workgroupSize[d] = newLocalWorkSize[d];
    }

    UR_CALL(setMutableGroupSizeDesc(mutableGroupSizeDesc, dim, workgroupSize,
                                    *nextDesc, command->commandId));
    *nextDesc = mutableGroupSizeDesc.get();
    descs.push_back(std::move(mutableGroupSizeDesc));
  }

  // Update global-size/group-count if provided, and also
  // local-size/group-size if required
  if (size_t *newGlobalWorkSize = commandDesc.pNewGlobalWorkSize;
      (newGlobalWorkSize || newLocalWorkSize) && dim > 0) {

    // If a new global work size is provided update that in the command,
    // otherwise the previous work group size will be used
    if (newGlobalWorkSize) {
      command->workDim = dim;
      command->setGlobalWorkSize(newGlobalWorkSize);
    }

    // If a new global work size is provided but a new local work size is not
    // then we still need to update local work size based on the size
    // suggested
    // by the driver for the kernel.
    bool updateWGSize = newLocalWorkSize == nullptr;

    ze_kernel_handle_t zeKernel{};
    UR_CALL(getZeKernel(command->kernel, zeKernel, device));

    uint32_t workgroupSize[3];

    UR_CALL(calculateKernelWorkDimensions(
        zeKernel, device, zeThreadGroupDimensionsList, workgroupSize, dim,
        command->globalWorkSize, newLocalWorkSize));

    auto mutableGroupCountDesc =
        std::make_unique<ZeStruct<ze_mutable_group_count_exp_desc_t>>();
    UR_CALL(setMutableGroupCountDesc(mutableGroupCountDesc,
                                     &zeThreadGroupDimensionsList, *nextDesc,
                                     command->commandId));
    *nextDesc = mutableGroupCountDesc.get();
    descs.push_back(std::move(mutableGroupCountDesc));

    if (updateWGSize) {
      auto mutableGroupSizeDesc =
          std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();
      UR_CALL(setMutableGroupSizeDesc(mutableGroupSizeDesc, dim, workgroupSize,
                                      *nextDesc, command->commandId));
      *nextDesc = mutableGroupSizeDesc.get();
      descs.push_back(std::move(mutableGroupSizeDesc));
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t updateKernelArguments(
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    ur_result_t (*getMemPtr)(ur_mem_handle_t,
                             const ur_kernel_arg_mem_obj_properties_t *,
                             char **&, ur_device_handle_t,
                             device_ptr_storage_t *),
    ur_device_handle_t device, desc_storage_t &descs,
    device_ptr_storage_t *ptrStorage) {
  for (uint32_t newMemObjArgNum = commandDesc.numNewMemObjArgs;
       newMemObjArgNum-- > 0;) {
    ur_exp_command_buffer_update_memobj_arg_desc_t newMemObjArgDesc =
        commandDesc.pNewMemObjArgList[newMemObjArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    ur_mem_handle_t newMemObjArg = newMemObjArgDesc.hNewMemObjArg;

    // The newMemObjArg may be a NULL pointer in which case a NULL value is
    // used for the kernel argument declared as a pointer to global or
    // constant memory.
    char **zeHandlePtr = nullptr;
    if (newMemObjArg) {
      const ur_kernel_arg_mem_obj_properties_t *properties =
          newMemObjArgDesc.pProperties;

      getMemPtr(newMemObjArg, properties, zeHandlePtr, device, ptrStorage);
    }

    UR_CALL(setMutableMemObjArgDesc(zeMutableArgDesc, newMemObjArgDesc.argIndex,
                                    zeHandlePtr, *nextDesc,
                                    command->commandId));
    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }

  // Update pointer arguments if provided.
  for (uint32_t newPointerArgNum = commandDesc.numNewPointerArgs;
       newPointerArgNum-- > 0;) {
    ur_exp_command_buffer_update_pointer_arg_desc_t newPointerArgDesc =
        commandDesc.pNewPointerArgList[newPointerArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    UR_CALL(setMutablePointerArgDesc(zeMutableArgDesc, newPointerArgDesc,
                                     *nextDesc, command->commandId));

    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }

  // Update value arguments if provided.
  for (uint32_t newValueArgNum = commandDesc.numNewValueArgs;
       newValueArgNum-- > 0;) {
    ur_exp_command_buffer_update_value_arg_desc_t newValueArgDesc =
        commandDesc.pNewValueArgList[newValueArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    UR_CALL(setMutableValueArgDesc(zeMutableArgDesc, newValueArgDesc, *nextDesc,
                                   command->commandId));

    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }
  return UR_RESULT_SUCCESS;
}

} // namespace
ur_result_t updateKernelHandle(ur_kernel_handle_t NewKernel,
                               ur_result_t (*GetZeKernel)(ur_kernel_handle_t,
                                                          ze_kernel_handle_t &,
                                                          ur_device_handle_t),
                               ur_device_handle_t Device,
                               ur_platform_handle_t Platform,
                               ze_command_list_handle_t ZeCommandList,
                               kernel_command_handle *Command) {
  ze_kernel_handle_t KernelHandle{};
  ze_kernel_handle_t ZeNewKernel{};
  UR_CALL(GetZeKernel(NewKernel, ZeNewKernel, Device));

  KernelHandle = ZeNewKernel;
  if (!Platform->ZeMutableCmdListExt.LoaderExtension) {
    ZE2UR_CALL(zelLoaderTranslateHandle,
               (ZEL_HANDLE_KERNEL, ZeNewKernel, (void **)&KernelHandle));
  }

  ZE2UR_CALL(Platform->ZeMutableCmdListExt
                 .zexCommandListUpdateMutableCommandKernelsExp,
             (ZeCommandList, 1, &Command->commandId, &KernelHandle));
  // Set current kernel to be the new kernel
  Command->kernel = NewKernel;
  return UR_RESULT_SUCCESS;
}

ur_result_t updateCommandBufferUnlocked(
    ur_result_t (*GetZeKernel)(ur_kernel_handle_t, ze_kernel_handle_t &,
                               ur_device_handle_t),
    ur_result_t (*GetMemPtr)(ur_mem_handle_t,
                             const ur_kernel_arg_mem_obj_properties_t *,
                             char **&, ur_device_handle_t,
                             device_ptr_storage_t *),
    ze_command_list_handle_t ZeCommandList, ur_platform_handle_t Platform,
    ur_device_handle_t Device, device_ptr_storage_t *PtrStorage,
    uint32_t NumKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDescs) {

  // We need the created descriptors to live till the point when
  // zeCommandListUpdateMutableCommandsExp is called at the end of the
  // function.
  std::vector<std::variant<
      std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>,
      std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>>,
      std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>>,
      std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>>>>
      Descs;

  std::vector<ze_group_count_t> ZeThreadGroupDimensionsList(
      NumKernelUpdates, ze_group_count_t{1, 1, 1});
  void *NextDesc = nullptr; // Used for pointer chaining
  // Iterate over every UR update descriptor struct, which corresponds to
  // several L0 update descriptor structs.
  for (uint32_t i = 0; i < NumKernelUpdates; i++) {
    const auto &CommandDesc = CommandDescs[i];
    auto Command = static_cast<kernel_command_handle *>(CommandDesc.hCommand);

    std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Guard(
        Command->Mutex, Command->kernel->Mutex);

    ur_kernel_handle_t NewKernel = CommandDesc.hNewKernel;
    if (NewKernel && Command->kernel != NewKernel) {
      updateKernelHandle(NewKernel, GetZeKernel, Device, Platform,
                         ZeCommandList, Command);
    }

    updateKernelSizes(CommandDesc, Command, &NextDesc,
                      ZeThreadGroupDimensionsList[i], GetZeKernel, Device,
                      Descs);
    updateKernelArguments(CommandDesc, Command, &NextDesc, GetMemPtr, Device,
                          Descs, PtrStorage);
  }

  ZeStruct<ze_mutable_commands_exp_desc_t> MutableCommandDesc{};
  MutableCommandDesc.pNext = NextDesc;
  MutableCommandDesc.flags = 0;
  ZE2UR_CALL(
      Platform->ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp,
      (ZeCommandList, &MutableCommandDesc));

  return UR_RESULT_SUCCESS;
}

/**
 * Validates contents of the update command descriptions.
 * @param[in] CommandBuffer The command-buffer which is being updated.
 * @param[in] Device The device associated with the command-buffer.
 * @param[in] ZeDriverGlobalOffsetExtensionFound Whether the driver supports
 * global offset extension.
 * @param[in] CommandDescSize The number of command descriptions.
 * @param[in] CommandDescs The update command configurations.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t validateCommandDescUnlocked(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_device_handle_t Device,
    bool ZeDriverGlobalOffsetExtensionFound, size_t CommandDescSize,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDescs) {

  auto SupportedFeatures =
      Device->ZeDeviceMutableCmdListsProperties->mutableCommandFlags;
  UR_LOG(DEBUG, "Mutable features supported by device {}", SupportedFeatures);

  for (size_t i = 0; i < CommandDescSize; i++) {
    const auto &CommandDesc = CommandDescs[i];
    auto Command = static_cast<kernel_command_handle *>(CommandDesc.hCommand);
    UR_ASSERT(CommandBuffer == Command->commandBuffer,
              UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP);

    UR_ASSERT(!CommandDesc.hNewKernel ||
                  (SupportedFeatures &
                   ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    // Check if the provided new kernel is in the list of valid alternatives.
    if (CommandDesc.hNewKernel &&
        !Command->validKernelHandles.count(CommandDesc.hNewKernel)) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    if (CommandDesc.newWorkDim != Command->workDim &&
        (!CommandDesc.pNewGlobalWorkOffset ||
         !CommandDesc.pNewGlobalWorkSize)) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    // Check if new global offset is provided.
    size_t *NewGlobalWorkOffset = CommandDesc.pNewGlobalWorkOffset;
    UR_ASSERT(
        !NewGlobalWorkOffset ||
            (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET),
        UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    if (NewGlobalWorkOffset) {
      if (!ZeDriverGlobalOffsetExtensionFound) {
        UR_LOG(ERR, "No global offset extension found on this driver");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    }

    // Check if new group size is provided.
    size_t *NewLocalWorkSize = CommandDesc.pNewLocalWorkSize;
    UR_ASSERT(!NewLocalWorkSize ||
                  (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

    // Check if new global size is provided and we need to update group count.
    size_t *NewGlobalWorkSize = CommandDesc.pNewGlobalWorkSize;
    UR_ASSERT(!NewGlobalWorkSize ||
                  (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    UR_ASSERT(!(NewGlobalWorkSize && !NewLocalWorkSize) ||
                  (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

    UR_ASSERT(
        (!CommandDesc.numNewMemObjArgs && !CommandDesc.numNewPointerArgs &&
         !CommandDesc.numNewValueArgs) ||
            (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS),
        UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }
  return UR_RESULT_SUCCESS;
}

/**
 * Creates a new command handle to use in future updates to the command-buffer.
 * @param[in] CommandBuffer The CommandBuffer associated with the new command.
 * @param[in] ZeCommandList The CommandList associated with the new command.
 * @param[in] Kernel  The Kernel associated with the new command.
 * @param[in] WorkDim Dimensions of the kernel associated with the new command.
 * @param[in] GlobalWorkSize Global work size of the kernel associated with the
 * new command.
 * @param[in] NumKernelAlternatives Number of kernel alternatives.
 * @param[in] KernelAlternatives List of kernel alternatives.
 * @param[in] Platform The platform associated with the new command.
 * @param[in] GetZeKernel Function to get the ze kernel handle.
 * @param[out] Command The handle to the new command.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t createCommandHandleUnlocked(
    ur_exp_command_buffer_handle_t CommandBuffer,
    ze_command_list_handle_t ZeCommandList, ur_kernel_handle_t Kernel,
    uint32_t WorkDim, const size_t *GlobalWorkSize,
    uint32_t NumKernelAlternatives, ur_kernel_handle_t *KernelAlternatives,
    ur_platform_handle_t Platform,
    ur_result_t (*GetZeKernel)(ur_kernel_handle_t, ze_kernel_handle_t &,
                               ur_device_handle_t),
    ur_device_handle_t Device,
    std::unique_ptr<kernel_command_handle> &Command) {

  for (uint32_t i = 0; i < NumKernelAlternatives; ++i) {
    UR_ASSERT(KernelAlternatives[i] != Kernel, UR_RESULT_ERROR_INVALID_VALUE);
  }
  // If command-buffer is updatable then get command id which is going to be
  // used if command is updated in the future. This
  // zeCommandListGetNextCommandIdExp can be called only if the command is
  // updatable.
  uint64_t CommandId = 0;
  ZeStruct<ze_mutable_command_id_exp_desc_t> ZeMutableCommandDesc;
  ZeMutableCommandDesc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET;

  if (NumKernelAlternatives > 0) {
    ZeMutableCommandDesc.flags |=
        ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION;

    std::vector<ze_kernel_handle_t> KernelHandles(NumKernelAlternatives + 1,
                                                  nullptr);

    ze_kernel_handle_t ZeMainKernel{};
    UR_CALL(GetZeKernel(Kernel, ZeMainKernel, Device));

    if (Platform->ZeMutableCmdListExt.LoaderExtension) {
      KernelHandles[0] = ZeMainKernel;
    } else {
      // If the L0 loader is not aware of the MCL extension, the main kernel
      // handle needs to be translated.
      ZE2UR_CALL(zelLoaderTranslateHandle,
                 (ZEL_HANDLE_KERNEL, ZeMainKernel, (void **)&KernelHandles[0]));
    }

    for (size_t i = 0; i < NumKernelAlternatives; i++) {
      ze_kernel_handle_t ZeAltKernel{};
      UR_CALL(GetZeKernel(KernelAlternatives[i], ZeAltKernel, Device));

      if (Platform->ZeMutableCmdListExt.LoaderExtension) {
        KernelHandles[i + 1] = ZeAltKernel;
      } else {
        // If the L0 loader is not aware of the MCL extension, the kernel
        // alternatives need to be translated.
        ZE2UR_CALL(zelLoaderTranslateHandle, (ZEL_HANDLE_KERNEL, ZeAltKernel,
                                              (void **)&KernelHandles[i + 1]));
      }
    }

    ZE2UR_CALL(Platform->ZeMutableCmdListExt
                   .zexCommandListGetNextCommandIdWithKernelsExp,
               (ZeCommandList, &ZeMutableCommandDesc, NumKernelAlternatives + 1,
                KernelHandles.data(), &CommandId));

  } else {
    ZE2UR_CALL(Platform->ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp,
               (ZeCommandList, &ZeMutableCommandDesc, &CommandId));
  }

  try {
    Command = std::make_unique<kernel_command_handle>(
        CommandBuffer, Kernel, CommandId, WorkDim, NumKernelAlternatives,
        KernelAlternatives);

    Command->setGlobalWorkSize(GlobalWorkSize);

  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}
