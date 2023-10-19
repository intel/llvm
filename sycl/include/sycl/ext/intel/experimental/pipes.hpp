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
    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    if constexpr (std::is_same_v<_functionPropertiesT,
                                 oneapi::experimental::empty_properties_t>) {
      Success = !static_cast<bool>(
          __spirv_ReadPipe(_RPipe, &TempData, m_Size, m_Alignment));
    } else {
      detail::AnnotatedMemberValue<__ocl_RPipeTy<_dataT>, _functionPropertiesT>
          annotated_wrapper(_RPipe);
      Success = !static_cast<bool>(__spirv_ReadPipe(
          annotated_wrapper.MValue, &TempData, m_Size, m_Alignment));
    }
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
    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    if constexpr (std::is_same_v<_functionPropertiesT,
                                 oneapi::experimental::empty_properties_t>) {
      Success = !static_cast<bool>(
          __spirv_WritePipe(_WPipe, &Data, m_Size, m_Alignment));
    } else {
      detail::AnnotatedMemberValue<__ocl_WPipeTy<_dataT>, _functionPropertiesT>
          annotated_wrapper(_WPipe);
      Success = !static_cast<bool>(__spirv_WritePipe(
          annotated_wrapper.MValue, &Data, m_Size, m_Alignment));
    }
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
    __ocl_RPipeTy<_dataT> _RPipe =
        __spirv_CreatePipeFromPipeStorage_read<_dataT>(&m_Storage);
    _dataT TempData;
    if constexpr (std::is_same_v<_functionPropertiesT,
                                 oneapi::experimental::empty_properties_t>) {
      __spirv_ReadPipeBlockingINTEL(_RPipe, &TempData, m_Size, m_Alignment);
    } else {
      detail::AnnotatedMemberValue<__ocl_RPipeTy<_dataT>, _functionPropertiesT>
          annotated_wrapper(_RPipe);
      __spirv_ReadPipeBlockingINTEL(annotated_wrapper.MValue, &TempData, m_Size,
                                    m_Alignment);
    }
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
    __ocl_WPipeTy<_dataT> _WPipe =
        __spirv_CreatePipeFromPipeStorage_write<_dataT>(&m_Storage);
    if constexpr (std::is_same_v<_functionPropertiesT,
                                 oneapi::experimental::empty_properties_t>) {
      __spirv_WritePipeBlockingINTEL(_WPipe, &Data, m_Size, m_Alignment);
    } else {
      detail::AnnotatedMemberValue<__ocl_WPipeTy<_dataT>, _functionPropertiesT>
          annotated_wrapper(_WPipe);
      __spirv_WritePipeBlockingINTEL(annotated_wrapper.MValue, &Data, m_Size,
                                     m_Alignment);
    }
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
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl
