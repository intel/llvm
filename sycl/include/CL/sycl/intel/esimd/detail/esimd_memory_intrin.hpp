//==------------ esimd_memory_intrin.hpp - DPC++ Explicit SIMD API ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares Explicit SIMD intrinsics used to implement working with
// the SIMD classes objects.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/intel/esimd/detail/esimd_types.hpp>
#include <CL/sycl/intel/esimd/detail/esimd_util.hpp>
#include <CL/sycl/intel/esimd/esimd_enum.hpp>
#include <cstdint>

// flat_read does flat-address gather
template <typename Ty, int N, int NumBlk = 0,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<
    Ty, N * sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk)>
__esimd_flat_read(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                  int ElemsPerAddr = NumBlk,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL void
__esimd_flat_write(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                   sycl::intel::gpu::vector_type_t<
                       Ty, N * sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk)>
                       vals,
                   int ElemsPerAddr = NumBlk,
                   sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// flat_block_read reads a block of data from one flat address
template <typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_block_read_unaligned(uint64_t addr);

// flat_block_write writes a block of data using one flat address
template <typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL void
__esimd_flat_block_write(uint64_t addr,
                         sycl::intel::gpu::vector_type_t<Ty, N> vals);

// Reads a block of data from given surface at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_block_read(SurfIndAliasTy surf_ind, uint32_t offset);

// Writes given block of data to a surface with given index at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL void
__esimd_block_write(SurfIndAliasTy surf_ind, uint32_t offset,
                    sycl::intel::gpu::vector_type_t<Ty, N> vals);

// flat_read4 does flat-address gather4
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> SYCL_EXTERNAL
__esimd_flat_read4(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                   sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// flat_write does flat-address scatter
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL void __esimd_flat_write4(
    sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
    sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> vals,
    sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// flat_atomic: flat-address atomic
template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic0(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic1(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<Ty, N> src0,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H = sycl::intel::gpu::CacheHint::None,
          sycl::intel::gpu::CacheHint L3H = sycl::intel::gpu::CacheHint::None>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic2(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<Ty, N> src0,
                     sycl::intel::gpu::vector_type_t<Ty, N> src1,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

// esimd_barrier, generic group barrier
SYCL_EXTERNAL void __esimd_barrier();

// slm_fence sets the SLM read/write order
SYCL_EXTERNAL void __esimd_slm_fence(uint8_t cntl);

// slm_read does SLM gather
template <typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_read(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                 sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// slm_write does SLM scatter
template <typename Ty, int N>
SYCL_EXTERNAL void
__esimd_slm_write(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                  sycl::intel::gpu::vector_type_t<Ty, N> vals,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// slm_block_read reads a block of data from SLM
template <typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_block_read(uint32_t addr);

// slm_block_write writes a block of data to SLM
template <typename Ty, int N>
SYCL_EXTERNAL void
__esimd_slm_block_write(uint32_t addr,
                        sycl::intel::gpu::vector_type_t<Ty, N> vals);

// slm_read4 does SLM gather4
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)>
__esimd_slm_read4(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// slm_write4 does SLM scatter4
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask>
SYCL_EXTERNAL void __esimd_slm_write4(
    sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
    sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> vals,
    sycl::intel::gpu::vector_type_t<uint16_t, N> pred = 1);

// slm_atomic: SLM atomic
template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic0(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic1(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<Ty, N> src0,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic2(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<Ty, N> src0,
                    sycl::intel::gpu::vector_type_t<Ty, N> src1,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred);

// Media block load
//
// @param Ty the element data type.
//
// @param M the hight of the 2D block.
//
// @param N the width of the 2D block.
//
// @param TACC type of the surface handle.
//
// @param modifier top/bottom field surface access control.
//
// @param handle the surface handle.
//
// @param plane planar surface index.
//
// @param width the width of the return block.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @return the linearized 2D block data read from surface.
//
template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, M * N>
__esimd_media_block_load(unsigned modififer, TACC handle, unsigned plane,
                         unsigned width, unsigned x, unsigned y);

// Media block store
//
// @param Ty the element data type.
//
// @param M the hight of the 2D block.
//
// @param N the width of the 2D block.
//
// @param TACC type of the surface handle.
//
// @param modifier top/bottom field surface access control.
//
// @param handle the surface handle.
//
// @param plane planar surface index.
//
// @param width the width of the return block.
//
// @param x X-coordinate of the left upper rectangle corner in BYTES.
//
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @param vals the linearized 2D block data to be written to surface.
//
template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL void
__esimd_media_block_store(unsigned modififer, TACC handle, unsigned plane,
                          unsigned width, unsigned x, unsigned y,
                          sycl::intel::gpu::vector_type_t<Ty, M * N> vals);

#ifndef __SYCL_DEVICE_ONLY__

template <typename Ty, int N, int NumBlk, sycl::intel::gpu::CacheHint L1H,
          sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<
    Ty, N * sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk)>
__esimd_flat_read(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                  int ElemsPerAddr,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  auto NumBlkDecoded = sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk);
  sycl::intel::gpu::vector_type_t<
      Ty, N * sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk)>
      V;
  ElemsPerAddr = sycl::intel::gpu::ElemsPerAddrDecoding(ElemsPerAddr);

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) == 2)
        ElemsPerAddr = ElemsPerAddr / 2;
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          V[I * NumBlkDecoded + J] = *(Addr + J);
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          V[J * N + I] = *(Addr + J);
      }
    }
  }
  return V;
}

template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask,
          sycl::intel::gpu::CacheHint L1H, sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)>
__esimd_flat_read4(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                   sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> V;
  unsigned int Next = 0;

  if constexpr (HasR(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (HasG(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (HasB(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  if constexpr (HasA(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty) +
                                          sizeof(Ty));
        V[Next] = *Addr;
      }
    }
  }

  return V;
}

template <typename Ty, int N, int NumBlk, sycl::intel::gpu::CacheHint L1H,
          sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL void
__esimd_flat_write(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                   sycl::intel::gpu::vector_type_t<
                       Ty, N * sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk)>
                       vals,
                   int ElemsPerAddr,
                   sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  auto NumBlkDecoded = sycl::intel::gpu::ElemsPerAddrDecoding(NumBlk);
  ElemsPerAddr = sycl::intel::gpu::ElemsPerAddrDecoding(ElemsPerAddr);

  for (int I = 0; I < N; I++) {
    if (pred[I]) {
      Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
      if (sizeof(Ty) == 2)
        ElemsPerAddr = ElemsPerAddr / 2;
      if (sizeof(Ty) <= 2) {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          *(Addr + J) = vals[I * NumBlkDecoded + J];
      } else {
        for (int J = 0; J < NumBlkDecoded && J < ElemsPerAddr; J++)
          *(Addr + J) = vals[J * N + I];
      }
    }
  }
}

template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask,
          sycl::intel::gpu::CacheHint L1H, sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL void __esimd_flat_write4(
    sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
    sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> vals,
    sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> V;
  unsigned int Next = 0;

  if constexpr (HasR(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I]);
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (HasG(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (HasB(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }

  if constexpr (HasA(Mask)) {
    for (int I = 0; I < N; I++, Next++) {
      if (pred[I]) {
        Ty *Addr = reinterpret_cast<Ty *>(addrs[I] + sizeof(Ty) + sizeof(Ty) +
                                          sizeof(Ty));
        *Addr = vals[Next];
      }
    }
  }
}

template <typename Ty, int N, sycl::intel::gpu::CacheHint L1H,
          sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_block_read_unaligned(uint64_t addr) {
  sycl::intel::gpu::vector_type_t<Ty, N> V;

  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    V[I] = *Addr;
  }
  return V;
}

template <typename Ty, int N, sycl::intel::gpu::CacheHint L1H,
          sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL void
__esimd_flat_block_write(uint64_t addr,
                         sycl::intel::gpu::vector_type_t<Ty, N> vals) {
  for (int I = 0; I < N; I++) {
    Ty *Addr = reinterpret_cast<Ty *>(addr + I * sizeof(Ty));
    *Addr = vals[I];
  }
}

template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, M * N>
__esimd_media_block_load(unsigned modififer, TACC handle, unsigned plane,
                         unsigned width, unsigned x, unsigned y) {
  // On host the input surface is modeled as sycl image 2d object,
  // and the read/write access is done through accessor,
  // which is passed in as the handle argument.
  auto range = sycl::intel::gpu::AccessorPrivateProxy::getImageRange(handle);
  unsigned bpp = sycl::intel::gpu::AccessorPrivateProxy::getElemSize(handle);
  unsigned vpp = bpp / sizeof(Ty);
  unsigned int i = x / bpp;
  unsigned int j = y;

  assert(x % bpp == 0);
  unsigned int xbound = range[0] - 1;
  unsigned int ybound = range[1] - 1;

  sycl::intel::gpu::vector_type_t<Ty, M * N> vals;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col += vpp) {
      unsigned int xoff = (i > xbound) ? xbound : i;
      unsigned int yoff = (j > ybound) ? ybound : j;
      auto coords = cl::sycl::cl_int2(xoff, yoff);
      cl::sycl::cl_uint4 data = handle.read(coords);

      sycl::intel::gpu::vector_type_t<unsigned int, 4> res;
      for (int idx = 0; idx < 4; idx++) {
        res[idx] = data[idx];
      }

      constexpr int refN = sizeof(cl::sycl::cl_uint4) / sizeof(Ty);
      unsigned int stride = sizeof(cl::sycl::cl_uint4) / bpp;
      using refTy = sycl::intel::gpu::vector_type_t<Ty, refN>;
      auto ref = reinterpret_cast<refTy>(res);

      unsigned int offset1 = col + row * N;
      unsigned int offset2 = 0;
      for (int idx = 0; idx < vpp; idx++) {
        vals[offset1] = ref[offset2];
        offset1++;
        offset2 += stride;
      }
      i++;
    }
    i = x / bpp;
    j++;
  }

  return vals;
}

template <typename Ty, int M, int N, typename TACC>
SYCL_EXTERNAL void
__esimd_media_block_store(unsigned modififer, TACC handle, unsigned plane,
                          unsigned width, unsigned x, unsigned y,
                          sycl::intel::gpu::vector_type_t<Ty, M * N> vals) {
  unsigned bpp = sycl::intel::gpu::AccessorPrivateProxy::getElemSize(handle);
  unsigned vpp = bpp / sizeof(Ty);
  auto range = sycl::intel::gpu::AccessorPrivateProxy::getImageRange(handle);
  unsigned int i = x / bpp;
  unsigned int j = y;

  assert(x % bpp == 0);

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col += vpp) {
      constexpr int Sz = sizeof(cl::sycl::cl_uint4) / sizeof(Ty);
      sycl::intel::gpu::vector_type_t<Ty, Sz> res = 0;

      unsigned int offset1 = col + row * N;
      unsigned int offset2 = 0;
      unsigned int stride = sizeof(cl::sycl::cl_uint4) / bpp;
      for (int idx = 0; idx < vpp; idx++) {
        res[offset2] = vals[offset1];
        offset1++;
        offset2 += stride;
      }

      using refTy = sycl::intel::gpu::vector_type_t<unsigned int, 4>;
      auto ref = reinterpret_cast<refTy>(res);

      cl::sycl::cl_uint4 data;
      for (int idx = 0; idx < 4; idx++) {
        data[idx] = ref[idx];
      }

      if (i < range[0] && j < range[1]) {
        auto coords = cl::sycl::cl_int2(i, j);
        handle.write(coords, data);
      }
      i++;
    }
    i = x / bpp;
    j++;
  }
}

template <typename Ty, int N>
SYCL_EXTERNAL uint16_t __esimd_any(sycl::intel::gpu::vector_type_t<Ty, N> src) {
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] != 0)
      return 1;
  }
  return 0;
}

template <typename Ty, int N>
SYCL_EXTERNAL uint16_t __esimd_all(sycl::intel::gpu::vector_type_t<Ty, N> src) {
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] == 0)
      return 0;
  }
  return 1;
}

template <typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_dp4(sycl::intel::gpu::vector_type_t<Ty, N> v1,
            sycl::intel::gpu::vector_type_t<Ty, N> v2) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  for (auto i = 0; i != N; i += 4) {
    Ty dp = (v1[i] * v2[i]) + (v1[i + 1] * v2[i + 1]) +
            (v1[i + 2] * v2[i + 2]) + (v1[i + 3] * v2[i + 3]);
    retv[i] = dp;
    retv[i + 1] = dp;
    retv[i + 2] = dp;
    retv[i + 3] = dp;
  }
  return retv;
}

/// TODO
SYCL_EXTERNAL void __esimd_barrier() {}

SYCL_EXTERNAL void __esimd_slm_fence(uint8_t cntl) {}

template <typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_read(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                 sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

// slm_write does SLM scatter
template <typename Ty, int N>
SYCL_EXTERNAL void
__esimd_slm_write(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                  sycl::intel::gpu::vector_type_t<Ty, N> vals,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {}

// slm_block_read reads a block of data from SLM
template <typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_block_read(uint32_t addr) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

// slm_block_write writes a block of data to SLM
template <typename Ty, int N>
SYCL_EXTERNAL void
__esimd_slm_block_write(uint32_t addr,
                        sycl::intel::gpu::vector_type_t<Ty, N> vals) {}

// slm_read4 does SLM gather4
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)>
__esimd_slm_read4(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                  sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> retv;
  return retv;
}

// slm_write4 does SLM scatter4
template <typename Ty, int N, sycl::intel::gpu::ChannelMaskType Mask>
SYCL_EXTERNAL void __esimd_slm_write4(
    sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
    sycl::intel::gpu::vector_type_t<Ty, N * NumChannels(Mask)> vals,
    sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {}

// slm_atomic: SLM atomic
template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic0(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic1(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<Ty, N> src0,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_slm_atomic2(sycl::intel::gpu::vector_type_t<uint32_t, N> addrs,
                    sycl::intel::gpu::vector_type_t<Ty, N> src0,
                    sycl::intel::gpu::vector_type_t<Ty, N> src1,
                    sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H, sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic0(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H, sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic1(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<Ty, N> src0,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <sycl::intel::gpu::EsimdAtomicOpType Op, typename Ty, int N,
          sycl::intel::gpu::CacheHint L1H, sycl::intel::gpu::CacheHint L3H>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_flat_atomic2(sycl::intel::gpu::vector_type_t<uint64_t, N> addrs,
                     sycl::intel::gpu::vector_type_t<Ty, N> src0,
                     sycl::intel::gpu::vector_type_t<Ty, N> src1,
                     sycl::intel::gpu::vector_type_t<uint16_t, N> pred) {
  sycl::intel::gpu::vector_type_t<Ty, N> retv;
  return retv;
}

template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL sycl::intel::gpu::vector_type_t<Ty, N>
__esimd_block_read(SurfIndAliasTy surf_ind, uint32_t offset) {
  throw cl::sycl::feature_not_supported();
  return sycl::intel::gpu::vector_type_t<Ty, N>();
}

template <typename Ty, int N, typename SurfIndAliasTy>
SYCL_EXTERNAL void
__esimd_block_write(SurfIndAliasTy surf_ind, uint32_t offset,
                    sycl::intel::gpu::vector_type_t<Ty, N> vals) {

  throw cl::sycl::feature_not_supported();
}

#endif // __SYCL_DEVICE_ONLY__
