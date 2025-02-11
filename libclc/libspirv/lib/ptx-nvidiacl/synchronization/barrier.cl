//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

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
  unsigned int order = semantics & 0x1F;
  if (scope == Subgroup) {
    // use a full mask as barriers are required to be convergent and exited
    // threads can safely be in the mask
    __nvvm_bar_warp_sync(0xFFFFFFFF);
  } else if (scope == Device && memory == Device &&
             order == SequentiallyConsistent &&
             __clc_nvvm_reflect_arch() >= 700) {
    unsigned int env1, env2;
    __asm__ __volatile__("mov.u32 %0, %%envreg1;" : "=r"(env1));
    __asm__ __volatile__("mov.u32 %0, %%envreg2;" : "=r"(env2));
    long long envreg1 = env1;
    long long envreg2 = env2;
    // Bit field insert operation. Place 32 bits of envreg2 next to 32 bits of
    // envreg1: s64[envreg2][envreg1]. The resulting value is the address in
    // device global memory region, where atomic operations can be performed.
    long long atomicAddr;
    __asm__ __volatile__("bfi.b64 %0, %1, %2, 32, 32;"
                         : "=l"(atomicAddr)
                         : "l"(envreg1), "l"(envreg2));
    if (!atomicAddr) {
      __builtin_trap();
    } else {
      unsigned int tidX = __nvvm_read_ptx_sreg_tid_x();
      unsigned int tidY = __nvvm_read_ptx_sreg_tid_y();
      unsigned int tidZ = __nvvm_read_ptx_sreg_tid_z();
      if (tidX + tidY + tidZ == 0) {
        // Increment address by 4 to get the precise region initialized to 0.
        atomicAddr += 4;
        unsigned int nctaidX = __nvvm_read_ptx_sreg_nctaid_x();
        unsigned int nctaidY = __nvvm_read_ptx_sreg_nctaid_y();
        unsigned int nctaidZ = __nvvm_read_ptx_sreg_nctaid_z();
        unsigned int totalNctaid = nctaidX * nctaidY * nctaidZ;

        // Do atomic.add(1) for each CTA and spin ld.acquire in a loop until all
        // CTAs have performed the addition
        unsigned int prev, current;
        __asm__ __volatile__("atom.add.release.gpu.u32 %0,[%1],1;"
                             : "=r"(prev)
                             : "l"(atomicAddr));
        do {
          __asm__ __volatile__("ld.acquire.gpu.u32 %0,[%1];"
                               : "=r"(current)
                               : "l"(atomicAddr));
        } while (current % totalNctaid != 0);
      }
      __nvvm_barrier_sync(0);
    }
  } else {
    __syncthreads();
  }
}
