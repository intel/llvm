//===- OCLUtil.cpp - OCL Utilities ----------------------------------------===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements OCL utility functions.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "oclutil"

#include "OCLUtil.h"
#include "SPIRVEntry.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace SPIRV;

namespace OCLUtil {

#ifndef SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#define SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE SPIRAS_Private
#endif

#ifndef SPIRV_QUEUE_T_ADDR_SPACE
#define SPIRV_QUEUE_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_EVENT_T_ADDR_SPACE
#define SPIRV_EVENT_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_AVC_INTEL_T_ADDR_SPACE
#define SPIRV_AVC_INTEL_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_CLK_EVENT_T_ADDR_SPACE
#define SPIRV_CLK_EVENT_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_SAMPLER_T_ADDR_SPACE
#define SPIRV_SAMPLER_T_ADDR_SPACE SPIRAS_Constant
#endif

#ifndef SPIRV_RESERVE_ID_T_ADDR_SPACE
#define SPIRV_RESERVE_ID_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif
// Excerpt from SPIR 2.0 spec.:
//   Pipe objects are represented using pointers to the opaque %opencl.pipe LLVM
//   structure type which reside in the global address space.
#ifndef SPIRV_PIPE_ADDR_SPACE
#define SPIRV_PIPE_ADDR_SPACE SPIRAS_Global
#endif
// Excerpt from SPIR 2.0 spec.:
//   Note: Images data types reside in global memory and hence should be marked
//   as such in the "kernel arg addr space" metadata.
#ifndef SPIRV_IMAGE_ADDR_SPACE
#define SPIRV_IMAGE_ADDR_SPACE SPIRAS_Global
#endif

} // namespace OCLUtil

///////////////////////////////////////////////////////////////////////////////
//
// Map definitions
//
///////////////////////////////////////////////////////////////////////////////

using namespace OCLUtil;
namespace SPIRV {

template <> void SPIRVMap<OCLMemFenceKind, MemorySemanticsMask>::init() {
  add(OCLMF_Local, MemorySemanticsWorkgroupMemoryMask);
  add(OCLMF_Global, MemorySemanticsCrossWorkgroupMemoryMask);
  add(OCLMF_Image, MemorySemanticsImageMemoryMask);
}

template <>
void SPIRVMap<OCLMemFenceExtendedKind, MemorySemanticsMask>::init() {
  add(OCLMFEx_Local, MemorySemanticsWorkgroupMemoryMask);
  add(OCLMFEx_Global, MemorySemanticsCrossWorkgroupMemoryMask);
  add(OCLMFEx_Local_Global, MemorySemanticsWorkgroupMemoryMask |
                                MemorySemanticsCrossWorkgroupMemoryMask);
  add(OCLMFEx_Image, MemorySemanticsImageMemoryMask);
  add(OCLMFEx_Image_Local,
      MemorySemanticsWorkgroupMemoryMask | MemorySemanticsImageMemoryMask);
  add(OCLMFEx_Image_Global,
      MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsImageMemoryMask);
  add(OCLMFEx_Image_Local_Global, MemorySemanticsWorkgroupMemoryMask |
                                      MemorySemanticsCrossWorkgroupMemoryMask |
                                      MemorySemanticsImageMemoryMask);
}

template <>
void SPIRVMap<OCLMemOrderKind, unsigned, MemorySemanticsMask>::init() {
  add(OCLMO_relaxed, MemorySemanticsMaskNone);
  add(OCLMO_acquire, MemorySemanticsAcquireMask);
  add(OCLMO_release, MemorySemanticsReleaseMask);
  add(OCLMO_acq_rel, MemorySemanticsAcquireReleaseMask);
  add(OCLMO_seq_cst, MemorySemanticsSequentiallyConsistentMask);
}

template <> void SPIRVMap<OCLScopeKind, Scope>::init() {
  add(OCLMS_work_item, ScopeInvocation);
  add(OCLMS_work_group, ScopeWorkgroup);
  add(OCLMS_device, ScopeDevice);
  add(OCLMS_all_svm_devices, ScopeCrossDevice);
  add(OCLMS_sub_group, ScopeSubgroup);
}

template <> void SPIRVMap<std::string, SPIRVGroupOperationKind>::init() {
  add("reduce", GroupOperationReduce);
  add("scan_inclusive", GroupOperationInclusiveScan);
  add("scan_exclusive", GroupOperationExclusiveScan);
  add("ballot_bit_count", GroupOperationReduce);
  add("ballot_inclusive_scan", GroupOperationInclusiveScan);
  add("ballot_exclusive_scan", GroupOperationExclusiveScan);
  add("non_uniform_reduce", GroupOperationReduce);
  add("non_uniform_scan_inclusive", GroupOperationInclusiveScan);
  add("non_uniform_scan_exclusive", GroupOperationExclusiveScan);
  add("non_uniform_reduce_logical", GroupOperationReduce);
  add("non_uniform_scan_inclusive_logical", GroupOperationInclusiveScan);
  add("non_uniform_scan_exclusive_logical", GroupOperationExclusiveScan);
  add("clustered_reduce", GroupOperationClusteredReduce);
}

template <> void SPIRVMap<std::string, SPIRVFPRoundingModeKind>::init() {
  add("rte", FPRoundingModeRTE);
  add("rtz", FPRoundingModeRTZ);
  add("rtp", FPRoundingModeRTP);
  add("rtn", FPRoundingModeRTN);
}

template <> void SPIRVMap<OclExt::Kind, std::string>::init() {
#define _SPIRV_OP(x) add(OclExt::x, #x);
  _SPIRV_OP(cl_images)
  _SPIRV_OP(cl_doubles)
  _SPIRV_OP(cl_khr_int64_base_atomics)
  _SPIRV_OP(cl_khr_int64_extended_atomics)
  _SPIRV_OP(cl_khr_fp16)
  _SPIRV_OP(cl_khr_gl_sharing)
  _SPIRV_OP(cl_khr_gl_event)
  _SPIRV_OP(cl_khr_d3d10_sharing)
  _SPIRV_OP(cl_khr_media_sharing)
  _SPIRV_OP(cl_khr_d3d11_sharing)
  _SPIRV_OP(cl_khr_global_int32_base_atomics)
  _SPIRV_OP(cl_khr_global_int32_extended_atomics)
  _SPIRV_OP(cl_khr_local_int32_base_atomics)
  _SPIRV_OP(cl_khr_local_int32_extended_atomics)
  _SPIRV_OP(cl_khr_byte_addressable_store)
  _SPIRV_OP(cl_khr_3d_image_writes)
  _SPIRV_OP(cl_khr_gl_msaa_sharing)
  _SPIRV_OP(cl_khr_depth_images)
  _SPIRV_OP(cl_khr_gl_depth_images)
  _SPIRV_OP(cl_khr_subgroups)
  _SPIRV_OP(cl_khr_mipmap_image)
  _SPIRV_OP(cl_khr_mipmap_image_writes)
  _SPIRV_OP(cl_khr_egl_event)
  _SPIRV_OP(cl_khr_srgb_image_writes)
  _SPIRV_OP(cl_khr_extended_bit_ops)
#undef _SPIRV_OP
}

template <> void SPIRVMap<OclExt::Kind, SPIRVCapabilityKind>::init() {
  add(OclExt::cl_images, CapabilityImageBasic);
  add(OclExt::cl_doubles, CapabilityFloat64);
  add(OclExt::cl_khr_int64_base_atomics, CapabilityInt64Atomics);
  add(OclExt::cl_khr_int64_extended_atomics, CapabilityInt64Atomics);
  add(OclExt::cl_khr_fp16, CapabilityFloat16);
  add(OclExt::cl_khr_subgroups, CapabilityGroups);
  add(OclExt::cl_khr_mipmap_image, CapabilityImageMipmap);
  add(OclExt::cl_khr_mipmap_image_writes, CapabilityImageMipmap);
  add(OclExt::cl_khr_extended_bit_ops, CapabilityBitInstructions);
}

/// Map OpenCL work functions to SPIR-V builtin variables.
template <> void SPIRVMap<std::string, SPIRVBuiltinVariableKind>::init() {
  add("get_work_dim", BuiltInWorkDim);
  add("get_global_size", BuiltInGlobalSize);
  add("get_global_id", BuiltInGlobalInvocationId);
  add("get_global_offset", BuiltInGlobalOffset);
  add("get_local_size", BuiltInWorkgroupSize);
  add("get_enqueued_local_size", BuiltInEnqueuedWorkgroupSize);
  add("get_local_id", BuiltInLocalInvocationId);
  add("get_num_groups", BuiltInNumWorkgroups);
  add("get_group_id", BuiltInWorkgroupId);
  add("get_global_linear_id", BuiltInGlobalLinearId);
  add("get_local_linear_id", BuiltInLocalInvocationIndex);
  // cl_khr_subgroups
  add("get_sub_group_size", BuiltInSubgroupSize);
  add("get_max_sub_group_size", BuiltInSubgroupMaxSize);
  add("get_num_sub_groups", BuiltInNumSubgroups);
  add("get_enqueued_num_sub_groups", BuiltInNumEnqueuedSubgroups);
  add("get_sub_group_id", BuiltInSubgroupId);
  add("get_sub_group_local_id", BuiltInSubgroupLocalInvocationId);
  // cl_khr_subgroup_ballot
  add("get_sub_group_eq_mask", BuiltInSubgroupEqMask);
  add("get_sub_group_ge_mask", BuiltInSubgroupGeMask);
  add("get_sub_group_gt_mask", BuiltInSubgroupGtMask);
  add("get_sub_group_le_mask", BuiltInSubgroupLeMask);
  add("get_sub_group_lt_mask", BuiltInSubgroupLtMask);
}

// Maps uniqued OCL builtin function name to SPIR-V op code.
// A uniqued OCL builtin function name may be different from the real
// OCL builtin function name. e.g. instead of atomic_min, atomic_umin
// is used for atomic_min with unsigned integer parameter.
// work_group_ and sub_group_ functions are unified as group_ functions
// except work_group_barrier.
class SPIRVInstruction;
template <> void SPIRVMap<std::string, Op, SPIRVInstruction>::init() {
#define _SPIRV_OP(x, y) add("atom_" #x, OpAtomic##y);
  // cl_khr_int64_base_atomics builtins
  _SPIRV_OP(add, IAdd)
  _SPIRV_OP(sub, ISub)
  _SPIRV_OP(xchg, Exchange)
  _SPIRV_OP(dec, IDecrement)
  _SPIRV_OP(inc, IIncrement)
  _SPIRV_OP(cmpxchg, CompareExchange)
  // cl_khr_int64_extended_atomics builtins
  _SPIRV_OP(min, SMin)
  _SPIRV_OP(max, SMax)
  _SPIRV_OP(and, And)
  _SPIRV_OP(or, Or)
  _SPIRV_OP(xor, Xor)
#undef _SPIRV_OP
#define _SPIRV_OP(x, y) add("atomic_" #x, Op##y);
  // CL 2.0 atomic builtins
  _SPIRV_OP(flag_test_and_set_explicit, AtomicFlagTestAndSet)
  _SPIRV_OP(flag_clear_explicit, AtomicFlagClear)
  _SPIRV_OP(load_explicit, AtomicLoad)
  _SPIRV_OP(store_explicit, AtomicStore)
  _SPIRV_OP(exchange_explicit, AtomicExchange)
  _SPIRV_OP(compare_exchange_strong_explicit, AtomicCompareExchange)
  _SPIRV_OP(compare_exchange_weak_explicit, AtomicCompareExchangeWeak)
  _SPIRV_OP(inc, AtomicIIncrement)
  _SPIRV_OP(dec, AtomicIDecrement)
  _SPIRV_OP(fetch_add_explicit, AtomicIAdd)
  _SPIRV_OP(fetch_sub_explicit, AtomicISub)
  _SPIRV_OP(fetch_umin_explicit, AtomicUMin)
  _SPIRV_OP(fetch_umax_explicit, AtomicUMax)
  _SPIRV_OP(fetch_min_explicit, AtomicSMin)
  _SPIRV_OP(fetch_max_explicit, AtomicSMax)
  _SPIRV_OP(fetch_and_explicit, AtomicAnd)
  _SPIRV_OP(fetch_or_explicit, AtomicOr)
  _SPIRV_OP(fetch_xor_explicit, AtomicXor)
#undef _SPIRV_OP
#define _SPIRV_OP(x, y) add(#x, Op##y);
  _SPIRV_OP(dot, Dot)
  _SPIRV_OP(async_work_group_copy, GroupAsyncCopy)
  _SPIRV_OP(async_work_group_strided_copy, GroupAsyncCopy)
  _SPIRV_OP(wait_group_events, GroupWaitEvents)
  _SPIRV_OP(isequal, FOrdEqual)
  _SPIRV_OP(isnotequal, FUnordNotEqual)
  _SPIRV_OP(isgreater, FOrdGreaterThan)
  _SPIRV_OP(isgreaterequal, FOrdGreaterThanEqual)
  _SPIRV_OP(isless, FOrdLessThan)
  _SPIRV_OP(islessequal, FOrdLessThanEqual)
  _SPIRV_OP(islessgreater, LessOrGreater)
  _SPIRV_OP(isordered, Ordered)
  _SPIRV_OP(isunordered, Unordered)
  _SPIRV_OP(isfinite, IsFinite)
  _SPIRV_OP(isinf, IsInf)
  _SPIRV_OP(isnan, IsNan)
  _SPIRV_OP(isnormal, IsNormal)
  _SPIRV_OP(signbit, SignBitSet)
  _SPIRV_OP(any, Any)
  _SPIRV_OP(all, All)
  _SPIRV_OP(popcount, BitCount)
  _SPIRV_OP(get_fence, GenericPtrMemSemantics)
  // CL 2.0 kernel enqueue builtins
  _SPIRV_OP(enqueue_marker, EnqueueMarker)
  _SPIRV_OP(enqueue_kernel, EnqueueKernel)
  _SPIRV_OP(get_kernel_sub_group_count_for_ndrange_impl,
            GetKernelNDrangeSubGroupCount)
  _SPIRV_OP(get_kernel_max_sub_group_size_for_ndrange_impl,
            GetKernelNDrangeMaxSubGroupSize)
  _SPIRV_OP(get_kernel_work_group_size_impl, GetKernelWorkGroupSize)
  _SPIRV_OP(get_kernel_preferred_work_group_size_multiple_impl,
            GetKernelPreferredWorkGroupSizeMultiple)
  _SPIRV_OP(retain_event, RetainEvent)
  _SPIRV_OP(release_event, ReleaseEvent)
  _SPIRV_OP(create_user_event, CreateUserEvent)
  _SPIRV_OP(is_valid_event, IsValidEvent)
  _SPIRV_OP(set_user_event_status, SetUserEventStatus)
  _SPIRV_OP(capture_event_profiling_info, CaptureEventProfilingInfo)
  _SPIRV_OP(get_default_queue, GetDefaultQueue)
  _SPIRV_OP(ndrange_1D, BuildNDRange)
  _SPIRV_OP(ndrange_2D, BuildNDRange)
  _SPIRV_OP(ndrange_3D, BuildNDRange)
  // Generic Address Space Casts
  _SPIRV_OP(to_global, GenericCastToPtrExplicit)
  _SPIRV_OP(to_local, GenericCastToPtrExplicit)
  _SPIRV_OP(to_private, GenericCastToPtrExplicit)
  // CL 2.0 pipe builtins
  _SPIRV_OP(read_pipe_2, ReadPipe)
  _SPIRV_OP(write_pipe_2, WritePipe)
  _SPIRV_OP(read_pipe_2_bl, ReadPipeBlockingINTEL)
  _SPIRV_OP(write_pipe_2_bl, WritePipeBlockingINTEL)
  _SPIRV_OP(read_pipe_4, ReservedReadPipe)
  _SPIRV_OP(write_pipe_4, ReservedWritePipe)
  _SPIRV_OP(reserve_read_pipe, ReserveReadPipePackets)
  _SPIRV_OP(reserve_write_pipe, ReserveWritePipePackets)
  _SPIRV_OP(commit_read_pipe, CommitReadPipe)
  _SPIRV_OP(commit_write_pipe, CommitWritePipe)
  _SPIRV_OP(is_valid_reserve_id, IsValidReserveId)
  _SPIRV_OP(group_reserve_read_pipe, GroupReserveReadPipePackets)
  _SPIRV_OP(group_reserve_write_pipe, GroupReserveWritePipePackets)
  _SPIRV_OP(group_commit_read_pipe, GroupCommitReadPipe)
  _SPIRV_OP(group_commit_write_pipe, GroupCommitWritePipe)
  _SPIRV_OP(get_pipe_num_packets_ro, GetNumPipePackets)
  _SPIRV_OP(get_pipe_num_packets_wo, GetNumPipePackets)
  _SPIRV_OP(get_pipe_max_packets_ro, GetMaxPipePackets)
  _SPIRV_OP(get_pipe_max_packets_wo, GetMaxPipePackets)
  // CL 2.0 workgroup builtins
  _SPIRV_OP(group_all, GroupAll)
  _SPIRV_OP(group_any, GroupAny)
  _SPIRV_OP(group_broadcast, GroupBroadcast)
  _SPIRV_OP(group_iadd, GroupIAdd)
  _SPIRV_OP(group_fadd, GroupFAdd)
  _SPIRV_OP(group_fmin, GroupFMin)
  _SPIRV_OP(group_umin, GroupUMin)
  _SPIRV_OP(group_smin, GroupSMin)
  _SPIRV_OP(group_fmax, GroupFMax)
  _SPIRV_OP(group_umax, GroupUMax)
  _SPIRV_OP(group_smax, GroupSMax)
  // CL image builtins
  _SPIRV_OP(SampledImage, SampledImage)
  _SPIRV_OP(ImageSampleExplicitLod, ImageSampleExplicitLod)
  _SPIRV_OP(read_image, ImageRead)
  _SPIRV_OP(write_image, ImageWrite)
  _SPIRV_OP(get_image_channel_data_type, ImageQueryFormat)
  _SPIRV_OP(get_image_channel_order, ImageQueryOrder)
  _SPIRV_OP(get_image_num_mip_levels, ImageQueryLevels)
  _SPIRV_OP(get_image_num_samples, ImageQuerySamples)
  // Intel Subgroups builtins
  _SPIRV_OP(intel_sub_group_shuffle, SubgroupShuffleINTEL)
  _SPIRV_OP(intel_sub_group_shuffle_down, SubgroupShuffleDownINTEL)
  _SPIRV_OP(intel_sub_group_shuffle_up, SubgroupShuffleUpINTEL)
  _SPIRV_OP(intel_sub_group_shuffle_xor, SubgroupShuffleXorINTEL)
  // Intel media_block_io builtins
  _SPIRV_OP(intel_sub_group_media_block_read, SubgroupImageMediaBlockReadINTEL)
  _SPIRV_OP(intel_sub_group_media_block_write,
            SubgroupImageMediaBlockWriteINTEL)
  // cl_khr_subgroup_non_uniform_vote
  _SPIRV_OP(group_elect, GroupNonUniformElect)
  _SPIRV_OP(group_non_uniform_all, GroupNonUniformAll)
  _SPIRV_OP(group_non_uniform_any, GroupNonUniformAny)
  _SPIRV_OP(group_non_uniform_all_equal, GroupNonUniformAllEqual)
  // cl_khr_subgroup_ballot
  _SPIRV_OP(group_non_uniform_broadcast, GroupNonUniformBroadcast)
  _SPIRV_OP(group_broadcast_first, GroupNonUniformBroadcastFirst)
  _SPIRV_OP(group_ballot, GroupNonUniformBallot)
  _SPIRV_OP(group_inverse_ballot, GroupNonUniformInverseBallot)
  _SPIRV_OP(group_ballot_bit_extract, GroupNonUniformBallotBitExtract)
  _SPIRV_OP(group_ballot_bit_count_iadd, GroupNonUniformBallotBitCount)
  _SPIRV_OP(group_ballot_find_lsb, GroupNonUniformBallotFindLSB)
  _SPIRV_OP(group_ballot_find_msb, GroupNonUniformBallotFindMSB)
  // cl_khr_subgroup_non_uniform_arithmetic
  _SPIRV_OP(group_non_uniform_iadd, GroupNonUniformIAdd)
  _SPIRV_OP(group_non_uniform_fadd, GroupNonUniformFAdd)
  _SPIRV_OP(group_non_uniform_imul, GroupNonUniformIMul)
  _SPIRV_OP(group_non_uniform_fmul, GroupNonUniformFMul)
  _SPIRV_OP(group_non_uniform_smin, GroupNonUniformSMin)
  _SPIRV_OP(group_non_uniform_umin, GroupNonUniformUMin)
  _SPIRV_OP(group_non_uniform_fmin, GroupNonUniformFMin)
  _SPIRV_OP(group_non_uniform_smax, GroupNonUniformSMax)
  _SPIRV_OP(group_non_uniform_umax, GroupNonUniformUMax)
  _SPIRV_OP(group_non_uniform_fmax, GroupNonUniformFMax)
  _SPIRV_OP(group_non_uniform_iand, GroupNonUniformBitwiseAnd)
  _SPIRV_OP(group_non_uniform_ior, GroupNonUniformBitwiseOr)
  _SPIRV_OP(group_non_uniform_ixor, GroupNonUniformBitwiseXor)
  _SPIRV_OP(group_non_uniform_logical_iand, GroupNonUniformLogicalAnd)
  _SPIRV_OP(group_non_uniform_logical_ior, GroupNonUniformLogicalOr)
  _SPIRV_OP(group_non_uniform_logical_ixor, GroupNonUniformLogicalXor)
  // cl_khr_subgroup_shuffle
  _SPIRV_OP(group_shuffle, GroupNonUniformShuffle)
  _SPIRV_OP(group_shuffle_xor, GroupNonUniformShuffleXor)
  // cl_khr_subgroup_shuffle_relative
  _SPIRV_OP(group_shuffle_up, GroupNonUniformShuffleUp)
  _SPIRV_OP(group_shuffle_down, GroupNonUniformShuffleDown)
  // cl_khr_extended_bit_ops
  _SPIRV_OP(bitfield_insert, BitFieldInsert)
  _SPIRV_OP(bitfield_extract_signed, BitFieldSExtract)
  _SPIRV_OP(bitfield_extract_unsigned, BitFieldUExtract)
  _SPIRV_OP(bit_reverse, BitReverse)
#undef _SPIRV_OP
}

template <> void SPIRVMap<std::string, Op, OCL12Builtin>::init() {
#define _SPIRV_OP(x, y) add(#x, Op##y);
  _SPIRV_OP(add, AtomicIAdd)
  _SPIRV_OP(sub, AtomicISub)
  _SPIRV_OP(xchg, AtomicExchange)
  _SPIRV_OP(cmpxchg, AtomicCompareExchange)
  _SPIRV_OP(inc, AtomicIIncrement)
  _SPIRV_OP(dec, AtomicIDecrement)
  _SPIRV_OP(min, AtomicSMin)
  _SPIRV_OP(max, AtomicSMax)
  _SPIRV_OP(umin, AtomicUMin)
  _SPIRV_OP(umax, AtomicUMax)
  _SPIRV_OP(and, AtomicAnd)
  _SPIRV_OP(or, AtomicOr)
  _SPIRV_OP(xor, AtomicXor)
#undef _SPIRV_OP
}

// SPV_INTEL_device_side_avc_motion_estimation extension builtins
class SPIRVSubgroupsAVCIntelInst;
template <> void SPIRVMap<std::string, Op, SPIRVSubgroupsAVCIntelInst>::init() {
  // Here is a workaround for a bug in the specification:
  // 'avc' missed in 'intel_sub_group_avc' prefix.
  add("intel_sub_group_ime_ref_window_size",
      OpSubgroupAvcImeRefWindowSizeINTEL);

#define _SPIRV_OP(x, y) add("intel_sub_group_avc_" #x, OpSubgroupAvc##y##INTEL);
  // Initialization phase functions
  _SPIRV_OP(ime_initialize, ImeInitialize)
  _SPIRV_OP(fme_initialize, FmeInitialize)
  _SPIRV_OP(bme_initialize, BmeInitialize)
  _SPIRV_OP(sic_initialize, SicInitialize)

  // Result and payload types conversion functions
  _SPIRV_OP(mce_convert_to_ime_payload, MceConvertToImePayload)
  _SPIRV_OP(mce_convert_to_ime_result, MceConvertToImeResult)
  _SPIRV_OP(mce_convert_to_ref_payload, MceConvertToRefPayload)
  _SPIRV_OP(mce_convert_to_ref_result, MceConvertToRefResult)
  _SPIRV_OP(mce_convert_to_sic_payload, MceConvertToSicPayload)
  _SPIRV_OP(mce_convert_to_sic_result, MceConvertToSicResult)
  _SPIRV_OP(ime_convert_to_mce_payload, ImeConvertToMcePayload)
  _SPIRV_OP(ime_convert_to_mce_result, ImeConvertToMceResult)
  _SPIRV_OP(ref_convert_to_mce_payload, RefConvertToMcePayload)
  _SPIRV_OP(ref_convert_to_mce_result, RefConvertToMceResult)
  _SPIRV_OP(sic_convert_to_mce_payload, SicConvertToMcePayload)
  _SPIRV_OP(sic_convert_to_mce_result, SicConvertToMceResult)
#undef _SPIRV_OP

// MCE instructions
#define _SPIRV_OP(x, y)                                                        \
  add("intel_sub_group_avc_mce_" #x, OpSubgroupAvcMce##y##INTEL);
  _SPIRV_OP(get_default_inter_base_multi_reference_penalty,
            GetDefaultInterBaseMultiReferencePenalty)
  _SPIRV_OP(set_inter_base_multi_reference_penalty,
            SetInterBaseMultiReferencePenalty)
  _SPIRV_OP(get_default_inter_shape_penalty, GetDefaultInterShapePenalty)
  _SPIRV_OP(set_inter_shape_penalty, SetInterShapePenalty)
  _SPIRV_OP(get_default_inter_direction_penalty,
            GetDefaultInterDirectionPenalty)
  _SPIRV_OP(set_inter_direction_penalty, SetInterDirectionPenalty)
  _SPIRV_OP(get_default_intra_luma_shape_penalty,
            GetDefaultIntraLumaShapePenalty)
  _SPIRV_OP(get_default_inter_motion_vector_cost_table,
            GetDefaultInterMotionVectorCostTable)
  _SPIRV_OP(get_default_high_penalty_cost_table, GetDefaultHighPenaltyCostTable)
  _SPIRV_OP(get_default_medium_penalty_cost_table,
            GetDefaultMediumPenaltyCostTable)
  _SPIRV_OP(get_default_low_penalty_cost_table, GetDefaultLowPenaltyCostTable)
  _SPIRV_OP(set_motion_vector_cost_function, SetMotionVectorCostFunction)
  _SPIRV_OP(get_default_intra_luma_mode_penalty, GetDefaultIntraLumaModePenalty)
  _SPIRV_OP(get_default_non_dc_luma_intra_penalty,
            GetDefaultNonDcLumaIntraPenalty)
  _SPIRV_OP(get_default_intra_chroma_mode_base_penalty,
            GetDefaultIntraChromaModeBasePenalty)
  _SPIRV_OP(set_ac_only_haar, SetAcOnlyHaar)
  _SPIRV_OP(set_source_interlaced_field_polarity,
            SetSourceInterlacedFieldPolarity)
  _SPIRV_OP(set_single_reference_interlaced_field_polarity,
            SetSingleReferenceInterlacedFieldPolarity)
  _SPIRV_OP(set_dual_reference_interlaced_field_polarities,
            SetDualReferenceInterlacedFieldPolarities)
  _SPIRV_OP(get_motion_vectors, GetMotionVectors)
  _SPIRV_OP(get_inter_distortions, GetInterDistortions)
  _SPIRV_OP(get_best_inter_distortion, GetBestInterDistortions)
  _SPIRV_OP(get_inter_major_shape, GetInterMajorShape)
  _SPIRV_OP(get_inter_minor_shapes, GetInterMinorShape)
  _SPIRV_OP(get_inter_directions, GetInterDirections)
  _SPIRV_OP(get_inter_motion_vector_count, GetInterMotionVectorCount)
  _SPIRV_OP(get_inter_reference_ids, GetInterReferenceIds)
  _SPIRV_OP(get_inter_reference_interlaced_field_polarities,
            GetInterReferenceInterlacedFieldPolarities)
#undef _SPIRV_OP

// IME instructions
#define _SPIRV_OP(x, y)                                                        \
  add("intel_sub_group_avc_ime_" #x, OpSubgroupAvcIme##y##INTEL);
  _SPIRV_OP(set_single_reference, SetSingleReference)
  _SPIRV_OP(set_dual_reference, SetDualReference)
  _SPIRV_OP(ref_window_size, RefWindowSize)
  _SPIRV_OP(adjust_ref_offset, AdjustRefOffset)
  _SPIRV_OP(set_max_motion_vector_count, SetMaxMotionVectorCount)
  _SPIRV_OP(set_unidirectional_mix_disable, SetUnidirectionalMixDisable)
  _SPIRV_OP(set_early_search_termination_threshold,
            SetEarlySearchTerminationThreshold)
  _SPIRV_OP(set_weighted_sad, SetWeightedSad)
  _SPIRV_OP(evaluate_with_single_reference, EvaluateWithSingleReference)
  _SPIRV_OP(evaluate_with_dual_reference, EvaluateWithDualReference)
  _SPIRV_OP(evaluate_with_single_reference_streamin,
            EvaluateWithSingleReferenceStreamin)
  _SPIRV_OP(evaluate_with_dual_reference_streamin,
            EvaluateWithDualReferenceStreamin)
  _SPIRV_OP(evaluate_with_single_reference_streamout,
            EvaluateWithSingleReferenceStreamout)
  _SPIRV_OP(evaluate_with_dual_reference_streamout,
            EvaluateWithDualReferenceStreamout)
  _SPIRV_OP(evaluate_with_single_reference_streaminout,
            EvaluateWithSingleReferenceStreaminout)
  _SPIRV_OP(evaluate_with_dual_reference_streaminout,
            EvaluateWithDualReferenceStreaminout)
  _SPIRV_OP(get_single_reference_streamin, GetSingleReferenceStreamin)
  _SPIRV_OP(get_dual_reference_streamin, GetDualReferenceStreamin)
  _SPIRV_OP(strip_single_reference_streamout, StripSingleReferenceStreamout)
  _SPIRV_OP(strip_dual_reference_streamout, StripDualReferenceStreamout)
  _SPIRV_OP(get_border_reached, GetBorderReached)
  _SPIRV_OP(get_truncated_search_indication, GetTruncatedSearchIndication)
  _SPIRV_OP(get_unidirectional_early_search_termination,
            GetUnidirectionalEarlySearchTermination)
  _SPIRV_OP(get_weighting_pattern_minimum_motion_vector,
            GetWeightingPatternMinimumMotionVector)
  _SPIRV_OP(get_weighting_pattern_minimum_distortion,
            GetWeightingPatternMinimumDistortion)
#undef _SPIRV_OP

#define _SPIRV_OP(x, y)                                                        \
  add("intel_sub_group_avc_ime_get_streamout_major_shape_" #x,                 \
      OpSubgroupAvcImeGetStreamout##y##INTEL);
  _SPIRV_OP(motion_vectors_single_reference,
            SingleReferenceMajorShapeMotionVectors)
  _SPIRV_OP(distortions_single_reference, SingleReferenceMajorShapeDistortions)
  _SPIRV_OP(reference_ids_single_reference,
            SingleReferenceMajorShapeReferenceIds)
  _SPIRV_OP(motion_vectors_dual_reference, DualReferenceMajorShapeMotionVectors)
  _SPIRV_OP(distortions_dual_reference, DualReferenceMajorShapeDistortions)
  _SPIRV_OP(reference_ids_dual_reference, DualReferenceMajorShapeReferenceIds)
#undef _SPIRV_OP

// REF instructions
#define _SPIRV_OP(x, y)                                                        \
  add("intel_sub_group_avc_ref_" #x, OpSubgroupAvcRef##y##INTEL);
  _SPIRV_OP(set_bidirectional_mix_disable, SetBidirectionalMixDisable)
  _SPIRV_OP(set_bilinear_filter_enable, SetBilinearFilterEnable)
  _SPIRV_OP(evaluate_with_single_reference, EvaluateWithSingleReference)
  _SPIRV_OP(evaluate_with_dual_reference, EvaluateWithDualReference)
  _SPIRV_OP(evaluate_with_multi_reference, EvaluateWithMultiReference)
  _SPIRV_OP(evaluate_with_multi_reference_interlaced,
            EvaluateWithMultiReferenceInterlaced)
#undef _SPIRV_OP

// SIC instructions
#define _SPIRV_OP(x, y)                                                        \
  add("intel_sub_group_avc_sic_" #x, OpSubgroupAvcSic##y##INTEL);
  _SPIRV_OP(configure_skc, ConfigureSkc)
  _SPIRV_OP(configure_ipe_luma, ConfigureIpeLuma)
  _SPIRV_OP(configure_ipe_luma_chroma, ConfigureIpeLumaChroma)
  _SPIRV_OP(get_motion_vector_mask, GetMotionVectorMask)
  _SPIRV_OP(set_intra_luma_shape_penalty, SetIntraLumaShapePenalty)
  _SPIRV_OP(set_intra_luma_mode_cost_function, SetIntraLumaModeCostFunction)
  _SPIRV_OP(set_intra_chroma_mode_cost_function, SetIntraChromaModeCostFunction)
  _SPIRV_OP(set_skc_bilinear_filter_enable, SetBilinearFilterEnable)
  _SPIRV_OP(set_skc_forward_transform_enable, SetSkcForwardTransformEnable)
  _SPIRV_OP(set_block_based_raw_skip_sad, SetBlockBasedRawSkipSad)
  _SPIRV_OP(evaluate_ipe, EvaluateIpe)
  _SPIRV_OP(evaluate_with_single_reference, EvaluateWithSingleReference)
  _SPIRV_OP(evaluate_with_dual_reference, EvaluateWithDualReference)
  _SPIRV_OP(evaluate_with_multi_reference, EvaluateWithMultiReference)
  _SPIRV_OP(evaluate_with_multi_reference_interlaced,
            EvaluateWithMultiReferenceInterlaced)
  _SPIRV_OP(get_ipe_luma_shape, GetIpeLumaShape)
  _SPIRV_OP(get_best_ipe_luma_distortion, GetBestIpeLumaDistortion)
  _SPIRV_OP(get_best_ipe_chroma_distortion, GetBestIpeChromaDistortion)
  _SPIRV_OP(get_packed_ipe_luma_modes, GetPackedIpeLumaModes)
  _SPIRV_OP(get_ipe_chroma_mode, GetIpeChromaMode)
  _SPIRV_OP(get_packed_skc_luma_count_threshold, GetPackedSkcLumaCountThreshold)
  _SPIRV_OP(get_packed_skc_luma_sum_threshold, GetPackedSkcLumaSumThreshold)
  _SPIRV_OP(get_inter_raw_sads, GetInterRawSads)
#undef _SPIRV_OP
}

template <> void SPIRVMap<std::string, Op, OCLOpaqueType>::init() {
  add("opencl.event_t", OpTypeEvent);
  add("opencl.pipe_t", OpTypePipe);
  add("opencl.clk_event_t", OpTypeDeviceEvent);
  add("opencl.reserve_id_t", OpTypeReserveId);
  add("opencl.queue_t", OpTypeQueue);
  add("opencl.sampler_t", OpTypeSampler);
}

template <> void LLVMSPIRVAtomicRmwOpCodeMap::init() {
  add(llvm::AtomicRMWInst::Xchg, OpAtomicExchange);
  add(llvm::AtomicRMWInst::Add, OpAtomicIAdd);
  add(llvm::AtomicRMWInst::Sub, OpAtomicISub);
  add(llvm::AtomicRMWInst::And, OpAtomicAnd);
  add(llvm::AtomicRMWInst::Or, OpAtomicOr);
  add(llvm::AtomicRMWInst::Xor, OpAtomicXor);
  add(llvm::AtomicRMWInst::Max, OpAtomicSMax);
  add(llvm::AtomicRMWInst::Min, OpAtomicSMin);
  add(llvm::AtomicRMWInst::UMax, OpAtomicUMax);
  add(llvm::AtomicRMWInst::UMin, OpAtomicUMin);
}

} // namespace SPIRV

///////////////////////////////////////////////////////////////////////////////
//
// Functions for getting builtin call info
//
///////////////////////////////////////////////////////////////////////////////

namespace OCLUtil {

AtomicWorkItemFenceLiterals getAtomicWorkItemFenceLiterals(CallInst *CI) {
  return std::make_tuple(getArgAsInt(CI, 0),
                         static_cast<OCLMemOrderKind>(getArgAsInt(CI, 1)),
                         static_cast<OCLScopeKind>(getArgAsInt(CI, 2)));
}

size_t getAtomicBuiltinNumMemoryOrderArgs(StringRef Name) {
  if (Name.startswith("atomic_compare_exchange"))
    return 2;
  return 1;
}

size_t getSPIRVAtomicBuiltinNumMemoryOrderArgs(Op OC) {
  if (OC == OpAtomicCompareExchange || OC == OpAtomicCompareExchangeWeak)
    return 2;
  return 1;
}

bool isComputeAtomicOCLBuiltin(StringRef DemangledName) {
  if (!DemangledName.startswith(kOCLBuiltinName::AtomicPrefix) &&
      !DemangledName.startswith(kOCLBuiltinName::AtomPrefix))
    return false;

  return llvm::StringSwitch<bool>(DemangledName)
      .EndsWith("add", true)
      .EndsWith("sub", true)
      .EndsWith("inc", true)
      .EndsWith("dec", true)
      .EndsWith("cmpxchg", true)
      .EndsWith("min", true)
      .EndsWith("max", true)
      .EndsWith("and", true)
      .EndsWith("or", true)
      .EndsWith("xor", true)
      .EndsWith("add_explicit", true)
      .EndsWith("sub_explicit", true)
      .EndsWith("or_explicit", true)
      .EndsWith("xor_explicit", true)
      .EndsWith("and_explicit", true)
      .EndsWith("min_explicit", true)
      .EndsWith("max_explicit", true)
      .Default(false);
}

BarrierLiterals getBarrierLiterals(CallInst *CI) {
  auto N = CI->getNumArgOperands();
  assert(N == 1 || N == 2);

  StringRef DemangledName;
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  if (!oclIsBuiltin(CI->getCalledFunction()->getName(), DemangledName)) {
    assert(0 &&
           "call must a builtin (work_group_barrier or sub_group_barrier)");
  }

  OCLScopeKind Scope = OCLMS_work_group;
  if (DemangledName == kOCLBuiltinName::SubGroupBarrier) {
    Scope = OCLMS_sub_group;
  }

  return std::make_tuple(getArgAsInt(CI, 0),
                         N == 1 ? OCLMS_work_group
                                : static_cast<OCLScopeKind>(getArgAsInt(CI, 1)),
                         Scope);
}

unsigned getExtOp(StringRef OrigName, StringRef GivenDemangledName) {
  std::string DemangledName{GivenDemangledName};
  if (DemangledName.empty() || !oclIsBuiltin(OrigName, GivenDemangledName))
    return ~0U;
  LLVM_DEBUG(dbgs() << "getExtOp: demangled name: " << DemangledName << '\n');
  OCLExtOpKind EOC;
  bool Found = OCLExtOpMap::rfind(DemangledName, &EOC);
  if (!Found) {
    std::string Prefix;
    switch (lastFuncParamType(OrigName)) {
    case ParamType::UNSIGNED:
      Prefix = "u_";
      break;
    case ParamType::SIGNED:
      Prefix = "s_";
      break;
    case ParamType::FLOAT:
      Prefix = "f";
      break;
    case ParamType::UNKNOWN:
      break;
    }
    Found = OCLExtOpMap::rfind(Prefix + DemangledName, &EOC);
  }
  if (Found)
    return EOC;
  else
    return ~0U;
}

///////////////////////////////////////////////////////////////////////////////
//
// Functions for getting module info
//
///////////////////////////////////////////////////////////////////////////////

unsigned encodeOCLVer(unsigned short Major, unsigned char Minor,
                      unsigned char Rev) {
  return (Major * 100 + Minor) * 1000 + Rev;
}

std::tuple<unsigned short, unsigned char, unsigned char>
decodeOCLVer(unsigned Ver) {
  unsigned short Major = Ver / 100000;
  unsigned char Minor = (Ver % 100000) / 1000;
  unsigned char Rev = Ver % 1000;
  return std::make_tuple(Major, Minor, Rev);
}

unsigned getOCLVersion(Module *M, bool AllowMulti) {
  NamedMDNode *NamedMD = M->getNamedMetadata(kSPIR2MD::OCLVer);
  if (!NamedMD)
    return 0;
  assert(NamedMD->getNumOperands() > 0 && "Invalid SPIR");
  if (!AllowMulti && NamedMD->getNumOperands() != 1)
    report_fatal_error("Multiple OCL version metadata not allowed");

  // If the module was linked with another module, there may be multiple
  // operands.
  auto GetVer = [=](unsigned I) {
    auto MD = NamedMD->getOperand(I);
    return std::make_pair(getMDOperandAsInt(MD, 0), getMDOperandAsInt(MD, 1));
  };
  auto Ver = GetVer(0);
  for (unsigned I = 1, E = NamedMD->getNumOperands(); I != E; ++I)
    if (Ver != GetVer(I))
      report_fatal_error("OCL version mismatch");

  return encodeOCLVer(Ver.first, Ver.second, 0);
}

void decodeMDNode(MDNode *N, unsigned &X, unsigned &Y, unsigned &Z) {
  if (N == NULL)
    return;
  X = getMDOperandAsInt(N, 0);
  Y = getMDOperandAsInt(N, 1);
  Z = getMDOperandAsInt(N, 2);
}

/// Encode LLVM type by SPIR-V execution mode VecTypeHint
unsigned encodeVecTypeHint(Type *Ty) {
  if (Ty->isHalfTy())
    return 4;
  if (Ty->isFloatTy())
    return 5;
  if (Ty->isDoubleTy())
    return 6;
  if (IntegerType *IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getIntegerBitWidth()) {
    case 8:
      return 0;
    case 16:
      return 1;
    case 32:
      return 2;
    case 64:
      return 3;
    default:
      llvm_unreachable("invalid integer type");
    }
  }
  if (FixedVectorType *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getNumElements();
    return Size << 16 | encodeVecTypeHint(EleTy);
  }
  llvm_unreachable("invalid type");
  return ~0U;
}

Type *decodeVecTypeHint(LLVMContext &C, unsigned Code) {
  unsigned VecWidth = Code >> 16;
  unsigned Scalar = Code & 0xFFFF;
  Type *ST = nullptr;
  switch (Scalar) {
  case 0:
  case 1:
  case 2:
  case 3:
    ST = IntegerType::get(C, 1 << (3 + Scalar));
    break;
  case 4:
    ST = Type::getHalfTy(C);
    break;
  case 5:
    ST = Type::getFloatTy(C);
    break;
  case 6:
    ST = Type::getDoubleTy(C);
    break;
  default:
    llvm_unreachable("Invalid vec type hint");
    return nullptr;
  }
  if (VecWidth < 1)
    return ST;
  return FixedVectorType::get(ST, VecWidth);
}

unsigned transVecTypeHint(MDNode *Node) {
  return encodeVecTypeHint(getMDOperandAsType(Node, 0));
}

SPIRAddressSpace getOCLOpaqueTypeAddrSpace(Op OpCode) {
  switch (OpCode) {
  case OpTypeQueue:
    return SPIRV_QUEUE_T_ADDR_SPACE;
  case OpTypeEvent:
    return SPIRV_EVENT_T_ADDR_SPACE;
  case OpTypeDeviceEvent:
    return SPIRV_CLK_EVENT_T_ADDR_SPACE;
  case OpTypeReserveId:
    return SPIRV_RESERVE_ID_T_ADDR_SPACE;
  case OpTypePipe:
  case OpTypePipeStorage:
    return SPIRV_PIPE_ADDR_SPACE;
  case OpTypeImage:
  case OpTypeSampledImage:
    return SPIRV_IMAGE_ADDR_SPACE;
  case OpConstantSampler:
  case OpTypeSampler:
    return SPIRV_SAMPLER_T_ADDR_SPACE;
  default:
    if (isSubgroupAvcINTELTypeOpCode(OpCode))
      return SPIRV_AVC_INTEL_T_ADDR_SPACE;
    assert(false && "No address space is determined for some OCL type");
    return SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE;
  }
}

static SPIR::TypeAttributeEnum mapAddrSpaceEnums(SPIRAddressSpace Addrspace) {
  switch (Addrspace) {
  case SPIRAS_Private:
    return SPIR::ATTR_PRIVATE;
  case SPIRAS_Global:
    return SPIR::ATTR_GLOBAL;
  case SPIRAS_Constant:
    return SPIR::ATTR_CONSTANT;
  case SPIRAS_Local:
    return SPIR::ATTR_LOCAL;
  case SPIRAS_Generic:
    return SPIR::ATTR_GENERIC;
  case SPIRAS_GlobalDevice:
    return SPIR::ATTR_GLOBAL_DEVICE;
  case SPIRAS_GlobalHost:
    return SPIR::ATTR_GLOBAL_HOST;
  default:
    llvm_unreachable("Invalid addrspace enum member");
  }
  return SPIR::ATTR_NONE;
}

SPIR::TypeAttributeEnum
getOCLOpaqueTypeAddrSpace(SPIR::TypePrimitiveEnum Prim) {
  switch (Prim) {
  case SPIR::PRIMITIVE_QUEUE_T:
    return mapAddrSpaceEnums(SPIRV_QUEUE_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_EVENT_T:
    return mapAddrSpaceEnums(SPIRV_EVENT_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_CLK_EVENT_T:
    return mapAddrSpaceEnums(SPIRV_CLK_EVENT_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_RESERVE_ID_T:
    return mapAddrSpaceEnums(SPIRV_RESERVE_ID_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_PIPE_RO_T:
  case SPIR::PRIMITIVE_PIPE_WO_T:
    return mapAddrSpaceEnums(SPIRV_PIPE_ADDR_SPACE);
  case SPIR::PRIMITIVE_IMAGE1D_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE3D_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE3D_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_RW_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_RW_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE3D_RW_T:
    return mapAddrSpaceEnums(SPIRV_IMAGE_ADDR_SPACE);
  default:
    llvm_unreachable("No address space is determined for a SPIR primitive");
  }
  return SPIR::ATTR_NONE;
}

// Fetch type of invoke function passed to device execution built-ins
static FunctionType *getBlockInvokeTy(Function *F, unsigned BlockIdx) {
  auto Params = F->getFunctionType()->params();
  PointerType *FuncPtr = cast<PointerType>(Params[BlockIdx]);
  return cast<FunctionType>(FuncPtr->getElementType());
}

class OCLBuiltinFuncMangleInfo : public SPIRV::BuiltinFuncMangleInfo {
public:
  OCLBuiltinFuncMangleInfo(Function *F) : F(F) {}
  OCLBuiltinFuncMangleInfo(ArrayRef<Type *> ArgTypes)
      : ArgTypes(ArgTypes.vec()) {}
  Type *getArgTy(unsigned I) { return F->getFunctionType()->getParamType(I); }
  void init(StringRef UniqName) override {
    // Make a local copy as we will modify the string in init function
    std::string TempStorage = UniqName.str();
    auto NameRef = StringRef(TempStorage);

    // Helper functions to erase substrings from NameRef (i.e. TempStorage)
    auto EraseSubstring = [&NameRef, &TempStorage](const std::string &ToErase) {
      size_t Pos = TempStorage.find(ToErase);
      if (Pos != std::string::npos) {
        TempStorage.erase(Pos, ToErase.length());
        // re-take StringRef as TempStorage was updated
        NameRef = StringRef(TempStorage);
      }
    };
    auto EraseSymbol = [&NameRef, &TempStorage](size_t Index) {
      TempStorage.erase(Index, 1);
      // re-take StringRef as TempStorage was updated
      NameRef = StringRef(TempStorage);
    };

    if (NameRef.startswith("async_work_group")) {
      addUnsignedArg(-1);
      setArgAttr(1, SPIR::ATTR_CONST);
    } else if (NameRef.startswith("printf"))
      setVarArg(1);
    else if (NameRef.startswith("write_imageui"))
      addUnsignedArg(2);
    else if (NameRef.equals("prefetch")) {
      addUnsignedArg(1);
      setArgAttr(0, SPIR::ATTR_CONST);
    } else if (NameRef.equals("get_kernel_work_group_size") ||
               NameRef.equals(
                   "get_kernel_preferred_work_group_size_multiple")) {
      assert(F && "lack of necessary information");
      const size_t BlockArgIdx = 0;
      FunctionType *InvokeTy = getBlockInvokeTy(F, BlockArgIdx);
      if (InvokeTy->getNumParams() > 1)
        setLocalArgBlock(BlockArgIdx);
    } else if (NameRef.equals("enqueue_kernel")) {
      assert(F && "lack of necessary information");
      setEnumArg(1, SPIR::PRIMITIVE_KERNEL_ENQUEUE_FLAGS_T);
      addUnsignedArg(3);
      setArgAttr(4, SPIR::ATTR_CONST);
      // If there are arguments other then block context then these are pointers
      // to local memory so this built-in must be mangled accordingly.
      const size_t BlockArgIdx = 6;
      FunctionType *InvokeTy = getBlockInvokeTy(F, BlockArgIdx);
      if (InvokeTy->getNumParams() > 1) {
        setLocalArgBlock(BlockArgIdx);
        addUnsignedArg(BlockArgIdx + 1);
        setVarArg(BlockArgIdx + 2);
      }
    } else if (NameRef.startswith("get_") || NameRef.equals("nan") ||
               NameRef.equals("mem_fence") || NameRef.startswith("shuffle")) {
      addUnsignedArg(-1);
      if (NameRef.startswith(kOCLBuiltinName::GetFence)) {
        setArgAttr(0, SPIR::ATTR_CONST);
        addVoidPtrArg(0);
      }
    } else if (NameRef.contains("barrier")) {
      addUnsignedArg(0);
      if (NameRef.equals("work_group_barrier") ||
          NameRef.equals("sub_group_barrier"))
        setEnumArg(1, SPIR::PRIMITIVE_MEMORY_SCOPE);
    } else if (NameRef.startswith("atomic_work_item_fence")) {
      addUnsignedArg(0);
      setEnumArg(1, SPIR::PRIMITIVE_MEMORY_ORDER);
      setEnumArg(2, SPIR::PRIMITIVE_MEMORY_SCOPE);
    } else if (NameRef.startswith("atom_")) {
      setArgAttr(0, SPIR::ATTR_VOLATILE);
      if (NameRef.endswith("_umax") || NameRef.endswith("_umin")) {
        addUnsignedArg(-1);
        // We need to remove u to match OpenCL C built-in function name
        EraseSymbol(5);
      }
    } else if (NameRef.startswith("atomic")) {
      setArgAttr(0, SPIR::ATTR_VOLATILE);
      if (NameRef.contains("_umax") || NameRef.contains("_umin")) {
        addUnsignedArg(-1);
        // We need to remove u to match OpenCL C built-in function name
        if (NameRef.contains("_fetch"))
          EraseSymbol(13);
        else
          EraseSymbol(7);
      }
      if (NameRef.contains("store_explicit") ||
          NameRef.contains("exchange_explicit") ||
          (NameRef.startswith("atomic_fetch") &&
           NameRef.contains("explicit"))) {
        setEnumArg(2, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(3, SPIR::PRIMITIVE_MEMORY_SCOPE);
      } else if (NameRef.contains("load_explicit") ||
                 (NameRef.startswith("atomic_flag") &&
                  NameRef.contains("explicit"))) {
        setEnumArg(1, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(2, SPIR::PRIMITIVE_MEMORY_SCOPE);
      } else if (NameRef.endswith("compare_exchange_strong_explicit") ||
                 NameRef.endswith("compare_exchange_weak_explicit")) {
        setEnumArg(3, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(4, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(5, SPIR::PRIMITIVE_MEMORY_SCOPE);
      }
      // Don't set atomic property to the first argument of 1.2 atomic
      // built-ins.
      if (!NameRef.endswith("xchg") && // covers _cmpxchg too
          (NameRef.contains("fetch") ||
           !(NameRef.endswith("_add") || NameRef.endswith("_sub") ||
             NameRef.endswith("_inc") || NameRef.endswith("_dec") ||
             NameRef.endswith("_min") || NameRef.endswith("_max") ||
             NameRef.endswith("_and") || NameRef.endswith("_or") ||
             NameRef.endswith("_xor")))) {
        addAtomicArg(0);
      }
    } else if (NameRef.startswith("uconvert_")) {
      addUnsignedArg(0);
      NameRef = NameRef.drop_front(1);
      UnmangledName.erase(0, 1);
    } else if (NameRef.startswith("s_")) {
      if (NameRef.equals("s_upsample"))
        addUnsignedArg(1);
      NameRef = NameRef.drop_front(2);
    } else if (NameRef.startswith("u_")) {
      addUnsignedArg(-1);
      NameRef = NameRef.drop_front(2);
    } else if (NameRef.equals("fclamp")) {
      NameRef = NameRef.drop_front(1);
    }
    // handle [read|write]pipe builtins (plus two i32 literal args
    // required by SPIR 2.0 provisional specification):
    else if (NameRef.equals("read_pipe_2") || NameRef.equals("write_pipe_2")) {
      // with 2 arguments (plus two i32 literals):
      // int read_pipe (read_only pipe gentype p, gentype *ptr)
      // int write_pipe (write_only pipe gentype p, const gentype *ptr)
      addVoidPtrArg(1);
      addUnsignedArg(2);
      addUnsignedArg(3);
      // OpenCL-like representation of blocking pipes
    } else if (NameRef.equals("read_pipe_2_bl") ||
               NameRef.equals("write_pipe_2_bl")) {
      // with 2 arguments (plus two i32 literals):
      // int read_pipe_bl (read_only pipe gentype p, gentype *ptr)
      // int write_pipe_bl (write_only pipe gentype p, const gentype *ptr)
      addVoidPtrArg(1);
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (NameRef.equals("read_pipe_4") ||
               NameRef.equals("write_pipe_4")) {
      // with 4 arguments (plus two i32 literals):
      // int read_pipe (read_only pipe gentype p, reserve_id_t reserve_id, uint
      // index, gentype *ptr) int write_pipe (write_only pipe gentype p,
      // reserve_id_t reserve_id, uint index, const gentype *ptr)
      addUnsignedArg(2);
      addVoidPtrArg(3);
      addUnsignedArg(4);
      addUnsignedArg(5);
    } else if (NameRef.contains("reserve_read_pipe") ||
               NameRef.contains("reserve_write_pipe")) {
      // process [|work_group|sub_group]reserve[read|write]pipe builtins
      addUnsignedArg(1);
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (NameRef.contains("commit_read_pipe") ||
               NameRef.contains("commit_write_pipe")) {
      // process [|work_group|sub_group]commit[read|write]pipe builtins
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (NameRef.equals("capture_event_profiling_info")) {
      addVoidPtrArg(2);
      setEnumArg(1, SPIR::PRIMITIVE_CLK_PROFILING_INFO);
    } else if (NameRef.equals("enqueue_marker")) {
      setArgAttr(2, SPIR::ATTR_CONST);
      addUnsignedArg(1);
    } else if (NameRef.startswith("vload")) {
      addUnsignedArg(0);
      setArgAttr(1, SPIR::ATTR_CONST);
    } else if (NameRef.startswith("vstore")) {
      addUnsignedArg(1);
    } else if (NameRef.startswith("ndrange_")) {
      addUnsignedArg(-1);
      if (NameRef[8] == '2' || NameRef[8] == '3') {
        setArgAttr(-1, SPIR::ATTR_CONST);
      }
    } else if (NameRef.contains("umax")) {
      addUnsignedArg(-1);
      EraseSymbol(NameRef.find("umax"));
    } else if (NameRef.contains("umin")) {
      addUnsignedArg(-1);
      EraseSymbol(NameRef.find("umin"));
    } else if (NameRef.contains("broadcast")) {
      addUnsignedArg(-1);
    } else if (NameRef.startswith(kOCLBuiltinName::SampledReadImage)) {
      NameRef.consume_front(kOCLBuiltinName::Sampled);
      addSamplerArg(1);
    } else if (NameRef.contains(kOCLSubgroupsAVCIntel::Prefix)) {
      if (NameRef.contains("evaluate_ipe"))
        addSamplerArg(1);
      else if (NameRef.contains("evaluate_with_single_reference"))
        addSamplerArg(2);
      else if (NameRef.contains("evaluate_with_multi_reference")) {
        addUnsignedArg(1);
        std::string PostFix = "_interlaced";
        if (NameRef.contains(PostFix)) {
          addUnsignedArg(2);
          addSamplerArg(3);
          EraseSubstring(PostFix);
        } else
          addSamplerArg(2);
      } else if (NameRef.contains("evaluate_with_dual_reference"))
        addSamplerArg(3);
      else if (NameRef.contains("fme_initialize"))
        addUnsignedArgs(0, 6);
      else if (NameRef.contains("bme_initialize"))
        addUnsignedArgs(0, 7);
      else if (NameRef.contains("set_inter_base_multi_reference_penalty") ||
               NameRef.contains("set_inter_shape_penalty") ||
               NameRef.contains("set_inter_direction_penalty"))
        addUnsignedArg(0);
      else if (NameRef.contains("set_motion_vector_cost_function"))
        addUnsignedArgs(0, 2);
      else if (NameRef.contains("interlaced_field_polarity"))
        addUnsignedArg(0);
      else if (NameRef.contains("interlaced_field_polarities"))
        addUnsignedArgs(0, 1);
      else if (NameRef.contains(kOCLSubgroupsAVCIntel::MCEPrefix)) {
        if (NameRef.contains("get_default"))
          addUnsignedArgs(0, 1);
      } else if (NameRef.contains(kOCLSubgroupsAVCIntel::IMEPrefix)) {
        if (NameRef.contains("initialize"))
          addUnsignedArgs(0, 2);
        else if (NameRef.contains("set_single_reference"))
          addUnsignedArg(1);
        else if (NameRef.contains("set_dual_reference"))
          addUnsignedArg(2);
        else if (NameRef.contains("set_weighted_sad") ||
                 NameRef.contains("set_early_search_termination_threshold"))
          addUnsignedArg(0);
        else if (NameRef.contains("adjust_ref_offset"))
          addUnsignedArgs(1, 3);
        else if (NameRef.contains("set_max_motion_vector_count") ||
                 NameRef.contains("get_border_reached"))
          addUnsignedArg(0);
        else if (NameRef.contains("shape_distortions") ||
                 NameRef.contains("shape_motion_vectors") ||
                 NameRef.contains("shape_reference_ids")) {
          if (NameRef.contains("single_reference")) {
            addUnsignedArg(1);
            EraseSubstring("_single_reference");
          } else if (NameRef.contains("dual_reference")) {
            addUnsignedArgs(1, 2);
            EraseSubstring("_dual_reference");
          }
        } else if (NameRef.contains("ref_window_size"))
          addUnsignedArg(0);
      } else if (NameRef.contains(kOCLSubgroupsAVCIntel::SICPrefix)) {
        if (NameRef.contains("initialize") ||
            NameRef.contains("set_intra_luma_shape_penalty"))
          addUnsignedArg(0);
        else if (NameRef.contains("configure_ipe")) {
          if (NameRef.contains("_luma")) {
            addUnsignedArgs(0, 6);
            EraseSubstring("_luma");
          }
          if (NameRef.contains("_chroma")) {
            addUnsignedArgs(7, 9);
            EraseSubstring("_chroma");
          }
        } else if (NameRef.contains("configure_skc"))
          addUnsignedArgs(0, 4);
        else if (NameRef.contains("set_skc")) {
          if (NameRef.contains("forward_transform_enable"))
            addUnsignedArg(0);
        } else if (NameRef.contains("set_block")) {
          if (NameRef.contains("based_raw_skip_sad"))
            addUnsignedArg(0);
        } else if (NameRef.contains("get_motion_vector_mask")) {
          addUnsignedArgs(0, 1);
        } else if (NameRef.contains("luma_mode_cost_function"))
          addUnsignedArgs(0, 2);
        else if (NameRef.contains("chroma_mode_cost_function"))
          addUnsignedArg(0);
      }
    } else if (NameRef.startswith("intel_sub_group_shuffle")) {
      if (NameRef.endswith("_down") || NameRef.endswith("_up"))
        addUnsignedArg(2);
      else
        addUnsignedArg(1);
    } else if (NameRef.startswith("intel_sub_group_block_write")) {
      // distinguish write to image and other data types as position
      // of uint argument is different though name is the same.
      auto *Arg0Ty = getArgTy(0);
      if (Arg0Ty->isPointerTy() &&
          Arg0Ty->getPointerElementType()->isIntegerTy()) {
        addUnsignedArg(0);
        addUnsignedArg(1);
      } else {
        addUnsignedArg(2);
      }
    } else if (NameRef.startswith("intel_sub_group_block_read")) {
      // distinguish read from image and other data types as position
      // of uint argument is different though name is the same.
      auto *Arg0Ty = getArgTy(0);
      if (Arg0Ty->isPointerTy() &&
          Arg0Ty->getPointerElementType()->isIntegerTy()) {
        setArgAttr(0, SPIR::ATTR_CONST);
        addUnsignedArg(0);
      }
    } else if (NameRef.startswith("intel_sub_group_media_block_write")) {
      addUnsignedArg(3);
    } else if (NameRef.startswith(kOCLBuiltinName::SubGroupPrefix)) {
      if (NameRef.contains("ballot")) {
        if (NameRef.contains("inverse") || NameRef.contains("bit_count") ||
            NameRef.contains("inclusive_scan") ||
            NameRef.contains("exclusive_scan") ||
            NameRef.contains("find_lsb") || NameRef.contains("find_msb"))
          addUnsignedArg(0);
        else if (NameRef.contains("bit_extract")) {
          addUnsignedArgs(0, 1);
        }
      } else if (NameRef.contains("shuffle") || NameRef.contains("clustered"))
        addUnsignedArg(1);
    } else if (NameRef.startswith("bitfield_insert")) {
      addUnsignedArgs(2, 3);
    } else if (NameRef.startswith("bitfield_extract_signed") ||
               NameRef.startswith("bitfield_extract_unsigned")) {
      addUnsignedArgs(1, 2);
    }

    // Store the final version of a function name
    UnmangledName = NameRef.str();
  }
  // Auxiliarry information, it is expected that it is relevant at the moment
  // the init method is called.
  Function *F;                  // SPIRV decorated function
  // TODO: ArgTypes argument should get removed once all SPV-IR related issues
  // are resolved
  std::vector<Type *> ArgTypes; // Arguments of OCL builtin
};

CallInst *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    AttributeList *Attrs) {
  OCLBuiltinFuncMangleInfo BtnInfo(CI->getCalledFunction());
  return mutateCallInst(M, CI, ArgMutate, &BtnInfo, Attrs);
}

Instruction *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate, AttributeList *Attrs,
    bool TakeFuncName) {
  OCLBuiltinFuncMangleInfo BtnInfo(CI->getCalledFunction());
  return mutateCallInst(M, CI, ArgMutate, RetMutate, &BtnInfo, Attrs,
                        TakeFuncName);
}

static std::pair<StringRef, StringRef>
getSrcAndDstElememntTypeName(BitCastInst *BIC) {
  if (!BIC)
    return std::pair<StringRef, StringRef>("", "");

  Type *SrcTy = BIC->getSrcTy();
  Type *DstTy = BIC->getDestTy();
  if (SrcTy->isPointerTy())
    SrcTy = SrcTy->getPointerElementType();
  if (DstTy->isPointerTy())
    DstTy = DstTy->getPointerElementType();
  auto SrcST = dyn_cast<StructType>(SrcTy);
  auto DstST = dyn_cast<StructType>(DstTy);
  if (!DstST || !DstST->hasName() || !SrcST || !SrcST->hasName())
    return std::pair<StringRef, StringRef>("", "");

  return std::make_pair(SrcST->getName(), DstST->getName());
}

bool isSamplerInitializer(Instruction *Inst) {
  BitCastInst *BIC = dyn_cast<BitCastInst>(Inst);
  auto Names = getSrcAndDstElememntTypeName(BIC);
  if (Names.second == getSPIRVTypeName(kSPIRVTypeName::Sampler) &&
      Names.first == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
    return true;

  return false;
}

bool isPipeStorageInitializer(Instruction *Inst) {
  BitCastInst *BIC = dyn_cast<BitCastInst>(Inst);
  auto Names = getSrcAndDstElememntTypeName(BIC);
  if (Names.second == getSPIRVTypeName(kSPIRVTypeName::PipeStorage) &&
      Names.first == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
    return true;

  return false;
}

bool isSpecialTypeInitializer(Instruction *Inst) {
  return isSamplerInitializer(Inst) || isPipeStorageInitializer(Inst);
}

bool isSamplerTy(Type *Ty) {
  auto PTy = dyn_cast<PointerType>(Ty);
  if (!PTy)
    return false;

  auto STy = dyn_cast<StructType>(PTy->getElementType());
  return STy && STy->hasName() && STy->getName() == kSPR2TypeName::Sampler;
}

bool isPipeOrAddressSpaceCastBI(const StringRef MangledName) {
  return MangledName == "write_pipe_2" || MangledName == "read_pipe_2" ||
         MangledName == "write_pipe_2_bl" || MangledName == "read_pipe_2_bl" ||
         MangledName == "write_pipe_4" || MangledName == "read_pipe_4" ||
         MangledName == "reserve_write_pipe" ||
         MangledName == "reserve_read_pipe" ||
         MangledName == "commit_write_pipe" ||
         MangledName == "commit_read_pipe" ||
         MangledName == "work_group_reserve_write_pipe" ||
         MangledName == "work_group_reserve_read_pipe" ||
         MangledName == "work_group_commit_write_pipe" ||
         MangledName == "work_group_commit_read_pipe" ||
         MangledName == "get_pipe_num_packets_ro" ||
         MangledName == "get_pipe_max_packets_ro" ||
         MangledName == "get_pipe_num_packets_wo" ||
         MangledName == "get_pipe_max_packets_wo" ||
         MangledName == "sub_group_reserve_write_pipe" ||
         MangledName == "sub_group_reserve_read_pipe" ||
         MangledName == "sub_group_commit_write_pipe" ||
         MangledName == "sub_group_commit_read_pipe" ||
         MangledName == "to_global" || MangledName == "to_local" ||
         MangledName == "to_private";
}

bool isEnqueueKernelBI(const StringRef MangledName) {
  return MangledName == "__enqueue_kernel_basic" ||
         MangledName == "__enqueue_kernel_basic_events" ||
         MangledName == "__enqueue_kernel_varargs" ||
         MangledName == "__enqueue_kernel_events_varargs";
}

bool isKernelQueryBI(const StringRef MangledName) {
  return MangledName == "__get_kernel_work_group_size_impl" ||
         MangledName == "__get_kernel_sub_group_count_for_ndrange_impl" ||
         MangledName == "__get_kernel_max_sub_group_size_for_ndrange_impl" ||
         MangledName == "__get_kernel_preferred_work_group_size_multiple_impl";
}

// isUnfusedMulAdd checks if we have the following (most common for fp
// contranction) pattern in LLVM IR:
//
//   %mul = fmul float %a, %b
//   %add = fadd float %mul, %c
//
// This pattern indicates that fp contraction could have been disabled by
// #pragma OPENCL FP_CONTRACT OFF. When contraction is enabled (by a pragma or
// by clang's -ffp-contract=fast), clang would generate:
//
//   %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
//
// or
//
//   %mul = fmul contract float %a, %b
//   %add = fadd contract float %mul, %c
//
// Note that optimizations may form an unfused fmuladd from fadd+load or
// fadd+call, so this check is quite restrictive (see the comment below).
//
bool isUnfusedMulAdd(BinaryOperator *B) {
  if (B->getOpcode() != Instruction::FAdd &&
      B->getOpcode() != Instruction::FSub)
    return false;

  if (B->hasAllowContract()) {
    // If this fadd or fsub itself has a contract flag, the operation can be
    // contracted regardless of the operands.
    return false;
  }

  // Otherwise, we cannot easily tell if the operation can be a candidate for
  // contraction or not. Consider the following cases:
  //
  //   %mul = alloca float
  //   %t1 = fmul float %a, %b
  //   store float* %mul, float %t
  //   %t2 = load %mul
  //   %r = fadd float %t2, %c
  //
  // LLVM IR does not allow %r to be contracted. However, after an optimization
  // it becomes a candidate for contraction if ContractionOFF is not set in
  // SPIR-V:
  //
  //   %t1 = fmul float %a, %b
  //   %r = fadd float %t1, %c
  //
  // To be on a safe side, we disallow everything that is even remotely similar
  // to fmul + fadd.
  return true;
}

std::string getIntelSubgroupBlockDataPostfix(unsigned ElementBitSize,
                                             unsigned VectorNumElements) {
  std::ostringstream OSS;
  switch (ElementBitSize) {
  case 8:
    OSS << "_uc";
    break;
  case 16:
    OSS << "_us";
    break;
  case 32:
    // Intentionally does nothing since _ui variant is only an alias.
    break;
  case 64:
    OSS << "_ul";
    break;
  default:
    llvm_unreachable(
        "Incorrect data bitsize for intel_subgroup_block builtins");
  }
  switch (VectorNumElements) {
  case 1:
    break;
  case 2:
  case 4:
  case 8:
    OSS << VectorNumElements;
    break;
  case 16:
    assert(ElementBitSize == 8 &&
           "16 elements vector allowed only for char builtins");
    OSS << VectorNumElements;
    break;
  default:
    llvm_unreachable(
        "Incorrect vector length for intel_subgroup_block builtins");
  }
  return OSS.str();
}

void insertImageNameAccessQualifier(SPIRVAccessQualifierKind Acc,
                                    std::string &Name) {
  std::string QName = rmap<std::string>(Acc);
  // transform: read_only -> ro, write_only -> wo, read_write -> rw
  QName = QName.substr(0, 1) + QName.substr(QName.find("_") + 1, 1) + "_";
  assert(!Name.empty() && "image name should not be empty");
  Name.insert(Name.size() - 1, QName);
}
} // namespace OCLUtil

Value *SPIRV::transOCLMemScopeIntoSPIRVScope(Value *MemScope,
                                             Optional<int> DefaultCase,
                                             Instruction *InsertBefore) {
  if (auto *C = dyn_cast<ConstantInt>(MemScope)) {
    return ConstantInt::get(
        C->getType(), map<Scope>(static_cast<OCLScopeKind>(C->getZExtValue())));
  }

  // If memory_scope is not a constant, then we have to insert dynamic mapping:
  return getOrCreateSwitchFunc(kSPIRVName::TranslateOCLMemScope, MemScope,
                               OCLMemScopeMap::getMap(), /* IsReverse */ false,
                               DefaultCase, InsertBefore);
}

Value *SPIRV::transOCLMemOrderIntoSPIRVMemorySemantics(
    Value *MemOrder, Optional<int> DefaultCase, Instruction *InsertBefore) {
  if (auto *C = dyn_cast<ConstantInt>(MemOrder)) {
    return ConstantInt::get(
        C->getType(), mapOCLMemSemanticToSPIRV(
                          0, static_cast<OCLMemOrderKind>(C->getZExtValue())));
  }

  return getOrCreateSwitchFunc(kSPIRVName::TranslateOCLMemOrder, MemOrder,
                               OCLMemOrderMap::getMap(), /* IsReverse */ false,
                               DefaultCase, InsertBefore);
}

Value *
SPIRV::transSPIRVMemoryScopeIntoOCLMemoryScope(Value *MemScope,
                                               Instruction *InsertBefore) {
  if (auto *C = dyn_cast<ConstantInt>(MemScope)) {
    return ConstantInt::get(C->getType(), rmap<OCLScopeKind>(static_cast<Scope>(
                                              C->getZExtValue())));
  }

  if (auto *CI = dyn_cast<CallInst>(MemScope)) {
    Function *F = CI->getCalledFunction();
    if (F && F->getName().equals(kSPIRVName::TranslateOCLMemScope)) {
      // In case the SPIR-V module was created from an OpenCL program by
      // *this* SPIR-V generator, we know that the value passed to
      // __translate_ocl_memory_scope is what we should pass to the
      // OpenCL builtin now.
      return CI->getArgOperand(0);
    }
  }

  return getOrCreateSwitchFunc(kSPIRVName::TranslateSPIRVMemScope, MemScope,
                               OCLMemScopeMap::getRMap(),
                               /* IsReverse */ true, None, InsertBefore);
}

Value *
SPIRV::transSPIRVMemorySemanticsIntoOCLMemoryOrder(Value *MemorySemantics,
                                                   Instruction *InsertBefore) {
  if (auto *C = dyn_cast<ConstantInt>(MemorySemantics)) {
    return ConstantInt::get(C->getType(),
                            mapSPIRVMemSemanticToOCL(C->getZExtValue()).second);
  }

  if (auto *CI = dyn_cast<CallInst>(MemorySemantics)) {
    Function *F = CI->getCalledFunction();
    if (F && F->getName().equals(kSPIRVName::TranslateOCLMemOrder)) {
      // In case the SPIR-V module was created from an OpenCL program by
      // *this* SPIR-V generator, we know that the value passed to
      // __translate_ocl_memory_order is what we should pass to the
      // OpenCL builtin now.
      return CI->getArgOperand(0);
    }
  }

  // SPIR-V MemorySemantics contains both OCL mem_fence_flags and mem_order and
  // therefore, we need to apply mask
  int Mask = MemorySemanticsMaskNone | MemorySemanticsAcquireMask |
             MemorySemanticsReleaseMask | MemorySemanticsAcquireReleaseMask |
             MemorySemanticsSequentiallyConsistentMask;
  return getOrCreateSwitchFunc(kSPIRVName::TranslateSPIRVMemOrder,
                               MemorySemantics, OCLMemOrderMap::getRMap(),
                               /* IsReverse */ true, None, InsertBefore, Mask);
}

Value *SPIRV::transSPIRVMemorySemanticsIntoOCLMemFenceFlags(
    Value *MemorySemantics, Instruction *InsertBefore) {
  if (auto *C = dyn_cast<ConstantInt>(MemorySemantics)) {
    return ConstantInt::get(C->getType(),
                            mapSPIRVMemSemanticToOCL(C->getZExtValue()).first);
  }

  // TODO: any possible optimizations?
  // SPIR-V MemorySemantics contains both OCL mem_fence_flags and mem_order and
  // therefore, we need to apply mask
  int Mask = MemorySemanticsWorkgroupMemoryMask |
             MemorySemanticsCrossWorkgroupMemoryMask |
             MemorySemanticsImageMemoryMask;
  return getOrCreateSwitchFunc(kSPIRVName::TranslateSPIRVMemFence,
                               MemorySemantics,
                               OCLMemFenceExtendedMap::getRMap(),
                               /* IsReverse */ true, None, InsertBefore, Mask);
}

void llvm::mangleOpenClBuiltin(const std::string &UniqName,
                               ArrayRef<Type *> ArgTypes,
                               std::string &MangledName) {
  OCLUtil::OCLBuiltinFuncMangleInfo BtnInfo(ArgTypes);
  MangledName = SPIRV::mangleBuiltin(UniqName, ArgTypes, &BtnInfo);
}
