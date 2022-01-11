//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "fpga_utils.hpp"
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

template <class _name, class _dataT, int32_t _min_capacity = 0> class pipe {
public:
  // Non-blocking pipes
  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  template <class... _Params> static _dataT read(bool &_Success) {
#ifdef __SYCL_DEVICE_ONLY__
    static constexpr auto _anchor_id =
        _GetValue<int32_t, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        _GetValue3<int32_t, type, int32_t, latency_constraint,
                   _Params...>::value;

    static constexpr int32_t _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int32_t _cycle = std::get<2>(_constraint);
    int32_t _type = 0; // Default: _control_type == type::none
    if constexpr (_control_type == type::exact) {
      _type = 1;
    } else if constexpr (_control_type == type::max) {
      _type = 2;
    } else if constexpr (_control_type == type::min) {
      _type = 3;
    }

    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    _Success = !static_cast<bool>(__latency_control_nb_read_wrapper(
        _RPipe, &TempData, _anchor_id, _target_anchor, _type, _cycle));
    return TempData;
#else
    (void)_Success;
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  template <class... _Params>
  static void write(const _dataT &_Data, bool &_Success) {
#ifdef __SYCL_DEVICE_ONLY__
    static constexpr auto _anchor_id =
        _GetValue<int32_t, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        _GetValue3<int32_t, type, int32_t, latency_constraint,
                   _Params...>::value;

    static constexpr int32_t _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int32_t _cycle = std::get<2>(_constraint);
    int32_t _type = 0; // Default: _control_type == type::none
    if constexpr (_control_type == type::exact) {
      _type = 1;
    } else if constexpr (_control_type == type::max) {
      _type = 2;
    } else if constexpr (_control_type == type::min) {
      _type = 3;
    }

    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    _Success = !static_cast<bool>(__latency_control_nb_write_wrapper(
        _WPipe, &_Data, _anchor_id, _target_anchor, _type, _cycle));
#else
    (void)_Success;
    (void)_Data;
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Blocking pipes
  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  template <class... _Params> static _dataT read() {
#ifdef __SYCL_DEVICE_ONLY__
    static constexpr auto _anchor_id =
        _GetValue<int32_t, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        _GetValue3<int32_t, type, int32_t, latency_constraint,
                   _Params...>::value;

    static constexpr int32_t _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int32_t _cycle = std::get<2>(_constraint);
    int32_t _type = 0; // Default: _control_type == type::none
    if constexpr (_control_type == type::exact) {
      _type = 1;
    } else if constexpr (_control_type == type::max) {
      _type = 2;
    } else if constexpr (_control_type == type::min) {
      _type = 3;
    }

    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    __latency_control_bl_read_wrapper(_RPipe, &TempData, _anchor_id,
                                      _target_anchor, _type, _cycle);
    return TempData;
#else
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  template <class... _Params> static void write(const _dataT &_Data) {
#ifdef __SYCL_DEVICE_ONLY__
    static constexpr auto _anchor_id =
        _GetValue<int32_t, latency_anchor_id, _Params...>::value;
    static constexpr auto _constraint =
        _GetValue3<int32_t, type, int32_t, latency_constraint,
                   _Params...>::value;

    static constexpr int32_t _target_anchor = std::get<0>(_constraint);
    static constexpr type _control_type = std::get<1>(_constraint);
    static constexpr int32_t _cycle = std::get<2>(_constraint);
    int32_t _type = 0; // Default: _control_type == type::none
    if constexpr (_control_type == type::exact) {
      _type = 1;
    } else if constexpr (_control_type == type::max) {
      _type = 2;
    } else if constexpr (_control_type == type::min) {
      _type = 3;
    }

    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    __latency_control_bl_write_wrapper(_WPipe, &_Data, _anchor_id,
                                       _target_anchor, _type, _cycle);
#else
    (void)_Data;
    assert(!"Pipes are not supported on a host device!");
#endif // __SYCL_DEVICE_ONLY__
  }

private:
  static constexpr int32_t m_Size = sizeof(_dataT);
  static constexpr int32_t m_Alignment = alignof(_dataT);
  static constexpr int32_t m_Capacity = _min_capacity;
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr struct ConstantPipeStorage m_Storage = {m_Size, m_Alignment,
                                                           m_Capacity};

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static int32_t
  __latency_control_nb_read_wrapper(__ocl_RPipeTy<_T> Pipe, _T *Data,
                                    int32_t AnchorID, int32_t TargetAnchor,
                                    int32_t Type, int32_t Cycle) {
    return __spirv_ReadPipe(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static int32_t
  __latency_control_nb_write_wrapper(__ocl_WPipeTy<_T> Pipe, const _T *Data,
                                     int32_t AnchorID, int32_t TargetAnchor,
                                     int32_t Type, int32_t Cycle) {
    return __spirv_WritePipe(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static void __latency_control_bl_read_wrapper(__ocl_RPipeTy<_T> Pipe,
                                                _T *Data, int32_t AnchorID,
                                                int32_t TargetAnchor,
                                                int32_t Type, int32_t Cycle) {
    return __spirv_ReadPipeBlockingINTEL(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static void
  __latency_control_bl_write_wrapper(__ocl_WPipeTy<_T> Pipe, const _T *Data,
                                     int32_t AnchorID, int32_t TargetAnchor,
                                     int32_t Type, int32_t Cycle) {
    return __spirv_WritePipeBlockingINTEL(Pipe, Data, m_Size, m_Alignment);
  }
#endif // __SYCL_DEVICE_ONLY__
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
