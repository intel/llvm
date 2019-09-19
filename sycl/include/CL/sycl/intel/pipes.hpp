//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace intel {

template <class name, class dataT, int32_t min_capacity = 0> class pipe {
public:
  // Non-blocking pipes
  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  static dataT read(bool &Success) {
#ifdef __SYCL_DEVICE_ONLY__
    RPipeTy<dataT> RPipe =
      __spirv_CreatePipeFromPipeStorage_read<dataT>(&m_Storage);
    dataT TempData;
    Success = !static_cast<bool>(
        __spirv_ReadPipe(RPipe, &TempData, m_Size, m_Alignment));
    return TempData;
#else
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  static void write(const dataT &Data, bool &Success) {
#ifdef __SYCL_DEVICE_ONLY__
    WPipeTy<dataT> WPipe =
      __spirv_CreatePipeFromPipeStorage_write<dataT>(&m_Storage);
    Success = !static_cast<bool>(
        __spirv_WritePipe(WPipe, &Data, m_Size, m_Alignment));
#else
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Blocking pipes
  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  static dataT read() {
#ifdef __SYCL_DEVICE_ONLY__
    RPipeTy<dataT> RPipe =
      __spirv_CreatePipeFromPipeStorage_read<dataT>(&m_Storage);
    dataT TempData;
    __spirv_ReadPipeBlockingINTEL(RPipe, &TempData, m_Size, m_Alignment);
    return TempData;
#else
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  static void write(const dataT &Data) {
#ifdef __SYCL_DEVICE_ONLY__
    WPipeTy<dataT> WPipe =
      __spirv_CreatePipeFromPipeStorage_write<dataT>(&m_Storage);
    __spirv_WritePipeBlockingINTEL(WPipe, &Data, m_Size, m_Alignment);
#else
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

private:
  static constexpr int32_t m_Size = sizeof(dataT);
  static constexpr int32_t m_Alignment = alignof(dataT);
  static constexpr int32_t m_Capacity = min_capacity;
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr struct ConstantPipeStorage m_Storage = {m_Size, m_Alignment,
                                                           m_Capacity};
#endif // __SYCL_DEVICE_ONLY__
};

} // namespace intel
} // namespace sycl
} // namespace cl
