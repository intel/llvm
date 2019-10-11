//==------- pi_functionoffsets.h - Plugin Interface Function Offsets ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// #define NameoftheAPI Offset_to_add.
// Map the names of PI APIs to the corresponding offsets. This offset is added
// to the address of the Function Pointer list(returned by the Plugin) to get
// the location of the corresponding API Function Pointer implemented by the
// Plugin. Eg: Plugin returns address 0x0000 for the Function List. To get the
// function pointer for function piQueueCreate, we add 13*(8) = 104 to 0x0000
// and access it. So (*0x0068) gives the function pointer to piQueueCreate
// implemented by the Plugin. Call is made using:
// (*0x0068)(context,device,properties,queue);

#define FUNCTION_PTR_SIZE sizeof(void (*)())

// Platform

#define piPlatformsGet_Offset 0
#define piPlatformGetInfo_Offset 1 * FUNCTION_PTR_SIZE
// Device
#define piDevicesGet_Offset 2 * FUNCTION_PTR_SIZE
#define piDeviceGetInfo_Offset 3 * FUNCTION_PTR_SIZE
#define piDevicePartition_Offset 4 * FUNCTION_PTR_SIZE
#define piDeviceRetain_Offset 5 * FUNCTION_PTR_SIZE
#define piDeviceRelease_Offset 6 * FUNCTION_PTR_SIZE
#define piextDeviceSelectBinary_Offset 7 * FUNCTION_PTR_SIZE
#define piextGetDeviceFunctionPointer_Offset 8 * FUNCTION_PTR_SIZE
// Context
#define piContextCreate_Offset 9 * FUNCTION_PTR_SIZE
#define piContextGetInfo_Offset 10 * FUNCTION_PTR_SIZE
#define piContextRetain_Offset 11 * FUNCTION_PTR_SIZE
#define piContextRelease_Offset 12 * FUNCTION_PTR_SIZE
// Queue
#define piQueueCreate_Offset 13 * FUNCTION_PTR_SIZE
#define piQueueGetInfo_Offset 14 * FUNCTION_PTR_SIZE
#define piQueueFinish_Offset 15 * FUNCTION_PTR_SIZE
#define piQueueRetain_Offset 16 * FUNCTION_PTR_SIZE
#define piQueueRelease_Offset 17 * FUNCTION_PTR_SIZE
// Memory
#define piMemBufferCreate_Offset 18 * FUNCTION_PTR_SIZE
#define piMemImageCreate_Offset 19 * FUNCTION_PTR_SIZE
#define piMemGetInfo_Offset 20 * FUNCTION_PTR_SIZE
#define piMemImageGetInfo_Offset 21 * FUNCTION_PTR_SIZE
#define piMemRetain_Offset 22 * FUNCTION_PTR_SIZE
#define piMemRelease_Offset 23 * FUNCTION_PTR_SIZE
#define piMemBufferPartition_Offset 24 * FUNCTION_PTR_SIZE
// Program
#define piProgramCreate_Offset 25 * FUNCTION_PTR_SIZE
#define piclProgramCreateWithSource_Offset 26 * FUNCTION_PTR_SIZE
#define piclProgramCreateWithBinary_Offset 27 * FUNCTION_PTR_SIZE
#define piProgramGetInfo_Offset 28 * FUNCTION_PTR_SIZE
#define piProgramCompile_Offset 29 * FUNCTION_PTR_SIZE
#define piProgramBuild_Offset 30 * FUNCTION_PTR_SIZE
#define piProgramLink_Offset 31 * FUNCTION_PTR_SIZE
#define piProgramGetBuildInfo_Offset 32 * FUNCTION_PTR_SIZE
#define piProgramRetain_Offset 33 * FUNCTION_PTR_SIZE
#define piProgramRelease_Offset 34 * FUNCTION_PTR_SIZE
// Kernel
#define piKernelCreate_Offset 35 * FUNCTION_PTR_SIZE
#define piKernelSetArg_Offset 36 * FUNCTION_PTR_SIZE
#define piKernelGetInfo_Offset 37 * FUNCTION_PTR_SIZE
#define piKernelGetGroupInfo_Offset 38 * FUNCTION_PTR_SIZE
#define piKernelGetSubGroupInfo_Offset 39 * FUNCTION_PTR_SIZE
#define piKernelRetain_Offset 40 * FUNCTION_PTR_SIZE
#define piKernelRelease_Offset 41 * FUNCTION_PTR_SIZE
// Event
#define piEventCreate_Offset 42 * FUNCTION_PTR_SIZE
#define piEventGetInfo_Offset 43 * FUNCTION_PTR_SIZE
#define piEventGetProfilingInfo_Offset 44 * FUNCTION_PTR_SIZE
#define piEventsWait_Offset 45 * FUNCTION_PTR_SIZE
#define piEventSetCallback_Offset 46 * FUNCTION_PTR_SIZE
#define piEventSetStatus_Offset 47 * FUNCTION_PTR_SIZE
#define piEventRetain_Offset 48 * FUNCTION_PTR_SIZE
#define piEventRelease_Offset 49 * FUNCTION_PTR_SIZE
// Sampler
#define piSamplerCreate_Offset 50 * FUNCTION_PTR_SIZE
#define piSamplerGetInfo_Offset 51 * FUNCTION_PTR_SIZE
#define piSamplerRetain_Offset 52 * FUNCTION_PTR_SIZE
#define piSamplerRelease_Offset 53 * FUNCTION_PTR_SIZE
// Queue commands
#define piEnqueueKernelLaunch_Offset 54 * FUNCTION_PTR_SIZE
#define piEnqueueNativeKernel_Offset 55 * FUNCTION_PTR_SIZE
#define piEnqueueEventsWait_Offset 56 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferRead_Offset 57 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferReadRect_Offset 58 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferWrite_Offset 59 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferWriteRect_Offset 60 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferCopy_Offset 61 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferCopyRect_Offset 62 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferFill_Offset 63 * FUNCTION_PTR_SIZE
#define piEnqueueMemImageRead_Offset 64 * FUNCTION_PTR_SIZE
#define piEnqueueMemImageWrite_Offset 65 * FUNCTION_PTR_SIZE
#define piEnqueueMemImageCopy_Offset 66 * FUNCTION_PTR_SIZE
#define piEnqueueMemImageFill_Offset 67 * FUNCTION_PTR_SIZE
#define piEnqueueMemBufferMap_Offset 68 * FUNCTION_PTR_SIZE
#define piEnqueueMemUnmap_Offset 69 * FUNCTION_PTR_SIZE

