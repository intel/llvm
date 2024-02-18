//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/detail/export.hpp>                          // for __SYCL_EX...
#include <sycl/device.hpp>                                 // for device
#include <sycl/event.hpp>                                  // for event
#include <sycl/exception.hpp>                              // for make_erro...
#include <sycl/ext/intel/experimental/pipe_properties.hpp> // for protocol_...
#include <sycl/ext/oneapi/properties/properties.hpp>       // for ValueOrDe...
#include <sycl/handler.hpp>                                // for handler
#include <sycl/info/info_desc.hpp>                         // for event_com...
#include <sycl/memory_enums.hpp>                           // for memory_order
#include <sycl/queue.hpp>                                  // for queue

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/intel/experimental/fpga_utils.hpp>
#include <sycl/ext/oneapi/latency_control/properties.hpp>
#endif

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.hpp>
#endif

#include <stdint.h> // for int32_t
#include <string>   // for string
#include <tuple>    // for _Swallow_...

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

// A helper templateless base class to get the host_pipe name.
class pipe_base {

protected:
  pipe_base();
  ~pipe_base();

  __SYCL_EXPORT static std::string get_pipe_name(const void *HostPipePtr);
};

template <class _name, class _dataT, int32_t _min_capacity = 0,
          class _propertiesT = decltype(oneapi::experimental::properties{}),
          class = void>
class pipe : public pipe_base {
public:
  struct
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_attributes_global_variable(
          "sycl-host-pipe", "sycl-host-pipe-size", nullptr,
          sizeof(_dataT))]] [[__sycl_detail__::sycl_type(host_pipe)]]
#endif // __SYCL_DEVICE_ONLY___
      ConstantPipeStorageExp
#ifdef __SYCL_DEVICE_ONLY__
      : ConstantPipeStorage
#endif // __SYCL_DEVICE_ONLY___
  {
    int32_t _ReadyLatency;
    int32_t _BitsPerSymbol;
    bool _UsesValid;
    bool _FirstSymInHighOrderBits;
    protocol_name _Protocol;
  };

  // Non-blocking pipes

  // Host API
  static _dataT read(queue &Q, bool &Success,
                     memory_order Order = memory_order::seq_cst) {
    // Order is currently unused.
    std::ignore = Order;

    const device Dev = Q.get_device();
    bool IsPipeSupported =
        Dev.has_extension("cl_intel_program_scope_host_pipe");
    if (!IsPipeSupported) {
      return _dataT();
    }
    _dataT Data;
    void *DataPtr = &Data;
    const void *HostPipePtr = &m_Storage;
    const std::string PipeName = pipe_base::get_pipe_name(HostPipePtr);

    event E = Q.submit([=](handler &CGH) {
      CGH.ext_intel_read_host_pipe(PipeName, DataPtr,
                                   sizeof(_dataT) /* non-blocking */);
    });
    E.wait();
    if (E.get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete) {
      Success = true;
      return *(_dataT *)DataPtr;
    } else {
      Success = false;
      return _dataT();
    }
  }

  static void write(queue &Q, const _dataT &Data, bool &Success,
                    memory_order Order = memory_order::seq_cst) {
    // Order is currently unused.
    std::ignore = Order;

    const device Dev = Q.get_device();
    bool IsPipeSupported =
        Dev.has_extension("cl_intel_program_scope_host_pipe");
    if (!IsPipeSupported) {
      return;
    }

    const void *HostPipePtr = &m_Storage;
    const std::string PipeName = pipe_base::get_pipe_name(HostPipePtr);
    void *DataPtr = const_cast<_dataT *>(&Data);

    event E = Q.submit([=](handler &CGH) {
      CGH.ext_intel_write_host_pipe(PipeName, DataPtr,
                                    sizeof(_dataT) /* non-blocking */);
    });
    E.wait();
    Success = E.get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete;
  }

  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  template <typename _functionPropertiesT>
  static _dataT read(bool &Success, _functionPropertiesT) {
#ifdef __SYCL_DEVICE_ONLY__
    // Get latency control properties
    using _latency_anchor_id_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_anchor_id_key,
        detail::defaultLatencyAnchorIdProperty>::type;
    using _latency_constraint_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_constraint_key,
        detail::defaultLatencyConstraintProperty>::type;

    // Get latency control property values
    static constexpr int32_t _anchor_id = _latency_anchor_id_prop::value;
    static constexpr int32_t _target_anchor = _latency_constraint_prop::target;
    static constexpr latency_control_type _control_type =
        _latency_constraint_prop::type;
    static constexpr int32_t _relative_cycle = _latency_constraint_prop::cycle;

    int32_t _control_type_code = 0; // latency_control_type::none is default
    if constexpr (_control_type == latency_control_type::exact) {
      _control_type_code = 1;
    } else if constexpr (_control_type == latency_control_type::max) {
      _control_type_code = 2;
    } else if constexpr (_control_type == latency_control_type::min) {
      _control_type_code = 3;
    }

    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    Success = !static_cast<bool>(__latency_control_nb_read_wrapper(
        _RPipe, &TempData, _anchor_id, _target_anchor, _control_type_code,
        _relative_cycle));
    return TempData;
#else
    (void)Success;
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device-side API are not supported on a host device. Please use "
        "host-side API instead.");
#endif // __SYCL_DEVICE_ONLY__
  }

  static _dataT read(bool &Success) {
    return read(Success, oneapi::experimental::properties{});
  }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  template <typename _functionPropertiesT>
  static void write(const _dataT &Data, bool &Success, _functionPropertiesT) {
#ifdef __SYCL_DEVICE_ONLY__
    // Get latency control properties
    using _latency_anchor_id_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_anchor_id_key,
        detail::defaultLatencyAnchorIdProperty>::type;
    using _latency_constraint_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_constraint_key,
        detail::defaultLatencyConstraintProperty>::type;

    // Get latency control property values
    static constexpr int32_t _anchor_id = _latency_anchor_id_prop::value;
    static constexpr int32_t _target_anchor = _latency_constraint_prop::target;
    static constexpr latency_control_type _control_type =
        _latency_constraint_prop::type;
    static constexpr int32_t _relative_cycle = _latency_constraint_prop::cycle;

    int32_t _control_type_code = 0; // latency_control_type::none is default
    if constexpr (_control_type == latency_control_type::exact) {
      _control_type_code = 1;
    } else if constexpr (_control_type == latency_control_type::max) {
      _control_type_code = 2;
    } else if constexpr (_control_type == latency_control_type::min) {
      _control_type_code = 3;
    }

    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    Success = !static_cast<bool>(__latency_control_nb_write_wrapper(
        _WPipe, &Data, _anchor_id, _target_anchor, _control_type_code,
        _relative_cycle));
#else
    (void)Success;
    (void)Data;
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device-side API are not supported on a host device. Please use "
        "host-side API instead.");
#endif // __SYCL_DEVICE_ONLY__
  }

  static void write(const _dataT &Data, bool &Success) {
    write(Data, Success, oneapi::experimental::properties{});
  }

  static const void *get_host_ptr() { return &m_Storage; }

  // Blocking pipes

  // Host API
  static _dataT read(queue &Q, memory_order Order = memory_order::seq_cst) {
    // Order is currently unused.
    std::ignore = Order;

    const device Dev = Q.get_device();
    bool IsPipeSupported =
        Dev.has_extension("cl_intel_program_scope_host_pipe");
    if (!IsPipeSupported) {
      return _dataT();
    }
    _dataT Data;
    void *DataPtr = &Data;
    const void *HostPipePtr = &m_Storage;
    const std::string PipeName = pipe_base::get_pipe_name(HostPipePtr);
    event E = Q.submit([=](handler &CGH) {
      CGH.ext_intel_read_host_pipe(PipeName, DataPtr, sizeof(_dataT),
                                   true /*blocking*/);
    });
    E.wait();
    return *(_dataT *)DataPtr;
  }

  static void write(queue &Q, const _dataT &Data,
                    memory_order Order = memory_order::seq_cst) {
    // Order is currently unused.
    std::ignore = Order;

    const device Dev = Q.get_device();
    bool IsPipeSupported =
        Dev.has_extension("cl_intel_program_scope_host_pipe");
    if (!IsPipeSupported) {
      return;
    }
    const void *HostPipePtr = &m_Storage;
    const std::string PipeName = pipe_base::get_pipe_name(HostPipePtr);
    void *DataPtr = const_cast<_dataT *>(&Data);
    event E = Q.submit([=](handler &CGH) {
      CGH.ext_intel_write_host_pipe(PipeName, DataPtr, sizeof(_dataT),
                                    true /*blocking */);
    });
    E.wait();
  }

  // Reading from pipe is lowered to SPIR-V instruction OpReadPipe via SPIR-V
  // friendly LLVM IR.
  template <typename _functionPropertiesT>
  static _dataT read(_functionPropertiesT) {
#ifdef __SYCL_DEVICE_ONLY__
    // Get latency control properties
    using _latency_anchor_id_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_anchor_id_key,
        detail::defaultLatencyAnchorIdProperty>::type;
    using _latency_constraint_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_constraint_key,
        detail::defaultLatencyConstraintProperty>::type;

    // Get latency control property values
    static constexpr int32_t _anchor_id = _latency_anchor_id_prop::value;
    static constexpr int32_t _target_anchor = _latency_constraint_prop::target;
    static constexpr latency_control_type _control_type =
        _latency_constraint_prop::type;
    static constexpr int32_t _relative_cycle = _latency_constraint_prop::cycle;

    int32_t _control_type_code = 0; // latency_control_type::none is default
    if constexpr (_control_type == latency_control_type::exact) {
      _control_type_code = 1;
    } else if constexpr (_control_type == latency_control_type::max) {
      _control_type_code = 2;
    } else if constexpr (_control_type == latency_control_type::min) {
      _control_type_code = 3;
    }

    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    __latency_control_bl_read_wrapper(_RPipe, &TempData, _anchor_id,
                                      _target_anchor, _control_type_code,
                                      _relative_cycle);
    return TempData;
#else
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device-side API are not supported on a host device. Please use "
        "host-side API instead.");
#endif // __SYCL_DEVICE_ONLY__
  }

  static _dataT read() { return read(oneapi::experimental::properties{}); }

  // Writing to pipe is lowered to SPIR-V instruction OpWritePipe via SPIR-V
  // friendly LLVM IR.
  template <typename _functionPropertiesT>
  static void write(const _dataT &Data, _functionPropertiesT) {
#ifdef __SYCL_DEVICE_ONLY__
    // Get latency control properties
    using _latency_anchor_id_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_anchor_id_key,
        detail::defaultLatencyAnchorIdProperty>::type;
    using _latency_constraint_prop = typename detail::GetOrDefaultValT<
        _functionPropertiesT, latency_constraint_key,
        detail::defaultLatencyConstraintProperty>::type;

    // Get latency control property values
    static constexpr int32_t _anchor_id = _latency_anchor_id_prop::value;
    static constexpr int32_t _target_anchor = _latency_constraint_prop::target;
    static constexpr latency_control_type _control_type =
        _latency_constraint_prop::type;
    static constexpr int32_t _relative_cycle = _latency_constraint_prop::cycle;

    int32_t _control_type_code = 0; // latency_control_type::none is default
    if constexpr (_control_type == latency_control_type::exact) {
      _control_type_code = 1;
    } else if constexpr (_control_type == latency_control_type::max) {
      _control_type_code = 2;
    } else if constexpr (_control_type == latency_control_type::min) {
      _control_type_code = 3;
    }

    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    __latency_control_bl_write_wrapper(_WPipe, &Data, _anchor_id,
                                       _target_anchor, _control_type_code,
                                       _relative_cycle);
#else
    (void)Data;
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device-side API are not supported on a host device. Please use "
        "host-side API instead.");
#endif // __SYCL_DEVICE_ONLY__
  }

  static void write(const _dataT &Data) {
    write(Data, oneapi::experimental::properties{});
  }

private:
  static constexpr int32_t m_Size = sizeof(_dataT);
  static constexpr int32_t m_Alignment = alignof(_dataT);
  static constexpr int32_t m_Capacity = _min_capacity;

  static constexpr int32_t m_ready_latency =
      oneapi::experimental::detail::ValueOrDefault<
          _propertiesT, ready_latency_key>::template get<int32_t>(0);
  static constexpr int32_t m_bits_per_symbol =
      oneapi::experimental::detail::ValueOrDefault<
          _propertiesT, bits_per_symbol_key>::template get<int32_t>(8);
  static constexpr bool m_uses_valid =
      oneapi::experimental::detail::ValueOrDefault<
          _propertiesT, uses_valid_key>::template get<bool>(true);
  static constexpr bool m_first_symbol_in_high_order_bits =
      oneapi::experimental::detail::ValueOrDefault<
          _propertiesT,
          first_symbol_in_high_order_bits_key>::template get<int32_t>(0);
  static constexpr protocol_name m_protocol = oneapi::experimental::detail::
      ValueOrDefault<_propertiesT, protocol_key>::template get<protocol_name>(
          protocol_name::avalon_streaming_uses_ready);

public:
  static constexpr struct ConstantPipeStorageExp m_Storage = {
#ifdef __SYCL_DEVICE_ONLY__
      {m_Size, m_Alignment, m_Capacity},
#endif // __SYCL_DEVICE_ONLY___
      m_ready_latency,
      m_bits_per_symbol,
      m_uses_valid,
      m_first_symbol_in_high_order_bits,
      m_protocol};

#ifdef __SYCL_DEVICE_ONLY__
private:
  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static int32_t __latency_control_nb_read_wrapper(
      __ocl_RPipeTy<_T> Pipe, _T *Data, int32_t /* AnchorID */,
      int32_t /* TargetAnchor */, int32_t /* Type */, int32_t /* Cycle */) {
    return __spirv_ReadPipe(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static int32_t __latency_control_nb_write_wrapper(
      __ocl_WPipeTy<_T> Pipe, const _T *Data, int32_t /* AnchorID */,
      int32_t /* TargetAnchor */, int32_t /* Type */, int32_t /* Cycle */) {
    return __spirv_WritePipe(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static void __latency_control_bl_read_wrapper(
      __ocl_RPipeTy<_T> Pipe, _T *Data, int32_t /* AnchorID */,
      int32_t /* TargetAnchor */, int32_t /* Type */, int32_t /* Cycle */) {
    return __spirv_ReadPipeBlockingINTEL(Pipe, Data, m_Size, m_Alignment);
  }

  // FPGA BE will recognize this function and extract its arguments.
  // TODO: Pass latency control parameters via the __spirv_* builtin when ready.
  template <typename _T>
  static void __latency_control_bl_write_wrapper(
      __ocl_WPipeTy<_T> Pipe, const _T *Data, int32_t /* AnchorID*/,
      int32_t /* TargetAnchor */, int32_t /* Type */, int32_t /* Cycle */) {
    return __spirv_WritePipeBlockingINTEL(Pipe, Data, m_Size, m_Alignment);
  }
#endif // __SYCL_DEVICE_ONLY__
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl
