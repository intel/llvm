//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

int __clc_nvvm_reflect_arch();

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(unsigned int memory,
                                                  unsigned int semantics) {

  // for sm_70 and above membar becomes semantically identical to fence.sc.
  // However sm_70 and above also introduces a lightweight fence.acq_rel that
  // can be used to form either acquire or release strong operations.
  // Consult
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
  // for details.

  unsigned int order = semantics & 0x1F;
  if (__clc_nvvm_reflect_arch() < 700 ||
             order == SequentiallyConsistent) {
    if (memory == CrossDevice) {
      __nvvm_membar_sys();
    } else if (memory == Device) {
      __nvvm_membar_gl();
    } else {
      __nvvm_membar_cta();
    }
  } else if (order != None) {
    if (memory == CrossDevice) {
      __asm__ __volatile__("fence.acq_rel.sys;");
    } else if (memory == Device) {
      __asm__ __volatile__("fence.acq_rel.gpu;");
    } else {
      __asm__ __volatile__("fence.acq_rel.cta;");
    }
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(unsigned int scope, unsigned int memory,
                       unsigned int semantics) {
  if (scope == Subgroup) {
    // use a full mask as barriers are required to be convergent and exited
    // threads can safely be in the mask
    __nvvm_bar_warp_sync(0xFFFFFFFF);
  } else {
    __syncthreads();
  }
}
