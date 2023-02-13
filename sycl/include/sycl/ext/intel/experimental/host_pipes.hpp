// //==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
// //
// // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// // ===--------------------------------------------------------------------=== //

// #pragma once

// #include <sycl/context.hpp>
// #include <sycl/device.hpp>
// #include <sycl/ext/intel/experimental/pipe_properties.hpp>
// #include <sycl/ext/intel/experimental/host_pipe_base.hpp>
// #include <sycl/ext/oneapi/properties/properties.hpp>
// #include <sycl/queue.hpp>
// #include <type_traits>

// #ifdef XPTI_ENABLE_INSTRUMENTATION
// #include <xpti/xpti_data_types.h>
// #include <xpti/xpti_trace_framework.hpp>
// #endif

// namespace sycl {
// __SYCL_INLINE_VER_NAMESPACE(_V1) {
// namespace ext {
// namespace intel {
// namespace experimental {

// using default_pipe_properties =
//     decltype(sycl::ext::oneapi::experimental::properties(min_capacity<0>));

// template <class _name, class _dataT, class _propertiesT>
// class
// #ifdef __SYCL_DEVICE_ONLY__
//     [[__sycl_detail__::add_ir_attributes_global_variable("sycl-host-access",
//                                                          "readwrite")]]
// #endif
//     // TODO: change name to pipe, and merge into the existing pipe
//     // implementation
//     host_pipe : public host_pipe_base{

//   struct
// // Commented out since the host_pipe attribute is not introduced by the front
// // end yet. Confirm with Rob
// #ifdef __SYCL_DEVICE_ONLY__
//       [[__sycl_detail__::add_ir_attributes_global_variable(
//           "sycl-host-pipe",
//           nullptr)]] [[__sycl_detail__::
//                            host_pipe]] [[__sycl_detail__::
//                                              global_variable_allowed]] // may
//                                                                        // not be
//                                                                        // needed
// #endif
//       __pipeType {
//     const char __p;
//   };

//   static constexpr __pipeType __pipe = {0};

// public:
//   using value_type = _dataT;
//   static constexpr int32_t min_cap =
//       _propertiesT::template has_property<min_capacity_key>()
//           ? _propertiesT::template get_property<min_capacity_key>().value
//           : 0;

//   static const void *get_host_ptr() { return &__pipe; }

//   std::string get_pipe_name(const void *HostPipePtr){
//     return host_pipe_base::get_pipe_name(HostPipePtr);
//   }

//   // Blocking pipes
//   static _dataT read(queue & q, memory_order order = memory_order::seq_cst)
//   {
//      const device Dev = q.get_device();
//       bool IsReadPipeSupported =
//           Dev.has_extension("cl_intel_program_scope_host_pipe");
//       if (!IsReadPipeSupported) {
//         return &_dataT();
//       }
//       _dataT data;
//       const void *HostPipePtr = &__pipe;
//       const std::string pipe_name = host_pipe_base::get_pipe_name(HostPipePtr);
//       event e = q.submit([=](handler &CGH) {
//         CGH.read_write_host_pipe(pipe_name, (void *)(&data), sizeof(_dataT), false,
//                                 true /* read */);
//       });
//       e.wait();
//       return data;
//   }


//   static void write(queue & q, const _dataT &data,
//                     memory_order order = memory_order::seq_cst){
//   const device Dev = q.get_device();
//   bool IsReadPipeSupported =
//       Dev.has_extension("cl_intel_program_scope_host_pipe");
//   if (!IsReadPipeSupported) {
//     return;
//   }
//   const void *HostPipePtr = &__pipe;
//   const std::string pipe_name = host_pipe_base::get_pipe_name(HostPipePtr);
//   const void *data_ptr = &data;
//   event e = q.submit([=](handler &CGH) {
//     CGH.read_write_host_pipe(pipe_name, (void *)data_ptr, sizeof(_dataT), false,
//                              false /* write */);
//   });
//   e.wait();
// }


//   // Non-blocking pipes
//   static _dataT read(queue & q, bool &success_code,
//                      memory_order order = memory_order::seq_cst);
//   static void write(queue & q, const _dataT &data, bool &success_code,
//                     memory_order order = memory_order::seq_cst);

// private:
//   static constexpr int32_t m_Size = sizeof(_dataT);
//   static constexpr int32_t m_Alignment = alignof(_dataT);

// #ifdef __SYCL_DEVICE_ONLY__
//   static constexpr struct ConstantPipeStorage m_Storage = {m_Size, m_Alignment,
//                                                            min_cap};
// #endif // __SYCL_DEVICE_ONLY__
// };

// } // namespace experimental
// } // namespace intel
// } // namespace ext
// } // __SYCL_INLINE_VER_NAMESPACE(_V1)
// } // namespace sycl