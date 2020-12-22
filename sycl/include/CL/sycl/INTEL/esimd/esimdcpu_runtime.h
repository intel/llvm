//===--- esimdcpu_runtime.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ESIMDCPU_RUNTIME_H_INCLUDED_
#define _ESIMDCPU_RUNTIME_H_INCLUDED_

#include <vector>

#ifndef __GNUC__
#define ESIMD_API __declspec(dllexport)
#else // __GNUC__
#define ESIMD_API
#endif // __GNUC__

using fptrVoid = void (*)();

/// Imported from rt.h : Begin

/// Imported from rt.h : End

class ESimdCPUKernel {
private:
  const std::vector<uint32_t> m_singleGrpDim = {1, 1, 1};
  const std::vector<uint32_t> &m_spaceDim;
  uint32_t m_parallel;
  fptrVoid m_entryPoint;

public:
  ESIMD_API
  ESimdCPUKernel(fptrVoid entryPoint, const std::vector<uint32_t> &spaceDim);

  ESIMD_API
  void launchMT(const uint32_t argSize, const void *rawArg);
};

namespace cm_support {

ESIMD_API
int32_t thread_local_idx();

ESIMD_API
void mt_barrier();

ESIMD_API
void split_barrier(int flag);

ESIMD_API
void set_slm_size(size_t sz);

ESIMD_API
size_t get_slm_size();

ESIMD_API
char *get_slm();

ESIMD_API
void aux_barrier();

} // namespace cm_support

#endif // _ESIMDCPU_RUNTIME_H_INCLUDED_
