//==------------------ free_function_kernel_args.hpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Machinery for collecting the explicit arguments of a free function kernel
// that is submitted directly to a queue, bypassing the handler.
//
// The handler collects kernel arguments through its templated set_arg/set_args
// overloads, which store argument data in handler-owned storage and build a
// std::vector<detail::ArgDesc>. Those types (ArgDesc, CG::StorageInitHelper,
// KernelData) live entirely on the runtime (source) side and are not visible in
// the public headers. To collect arguments for a free function kernel without a
// handler we mirror the handler's split: the templated dispatch below runs in
// the header and forwards each argument through the ABI-exported, non-template
// methods of FreeFunctionArgCollector into a source-side object that owns the
// argument lifetime storage and produces the ArgDesc list.
//
// Scope: this path supports the argument kinds that can be launched without the
// scheduler resolving memory-object requirements, i.e. USM pointers, plain
// (device-copyable) data, work_group_memory, samplers and raw_kernel_arg. This
// mirrors the capability of the existing lambda-based direct submission path.
// Buffer/image accessors require requirement tracking and must be submitted
// through the handler overloads (e.g. nd_launch(handler &, ...)); passing them
// here is rejected at compile time.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/accessor.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {

// Only the address of a sampler is taken here; a forward declaration avoids
// pulling <sycl/sampler.hpp> (and its OpenCL interop dependencies) into every
// translation unit that includes this header.
class sampler;

namespace detail {

// Source-side object that owns the collected arguments plus the storage that
// keeps them alive. Defined in the runtime library.
class FreeFunctionArgsStorage;

/// Header-visible, ABI-stable collector for free function kernel arguments.
///
/// The templated setters below run in user code (the header) and translate each
/// argument into a call to one of the exported non-template methods, which
/// perform the same work handler::setArgHelper would (copy plain data into
/// owned storage, record the argument kind, etc.).
class __SYCL_EXPORT FreeFunctionArgCollector {
public:
  FreeFunctionArgCollector();
  ~FreeFunctionArgCollector();
  FreeFunctionArgCollector(FreeFunctionArgCollector &&) noexcept;
  FreeFunctionArgCollector &operator=(FreeFunctionArgCollector &&) noexcept;
  FreeFunctionArgCollector(const FreeFunctionArgCollector &) = delete;
  FreeFunctionArgCollector &operator=(const FreeFunctionArgCollector &) = delete;

  // Copies Size bytes at Ptr into owned storage and records an argument of the
  // given kind (kind_std_layout or kind_pointer) at ArgIndex.
  void addPlainArg(kernel_param_kind_t Kind, const void *Ptr, int Size,
                   int ArgIndex);

  // Copies the work_group_memory_impl pointed to by ImplPtr into owned storage
  // and records a work_group_memory argument referencing the copy. ImplPtr must
  // point to a sycl::detail::work_group_memory_impl object.
  void addWorkGroupMemoryArg(const void *ImplPtr, int ArgIndex);

  // Copies the sampler object at Ptr into owned storage and records it.
  void addSamplerArg(const void *Ptr, int ArgIndex);

  // Copies Size raw bytes into owned storage and records a std_layout argument.
  void addRawArg(const void *Ptr, size_t Size, int ArgIndex);

  // Transfers ownership of the underlying storage to the caller. After this the
  // collector is empty. Used to hand the collected arguments to the runtime.
  FreeFunctionArgsStorage *release() noexcept;

private:
  FreeFunctionArgsStorage *MImpl = nullptr;
};

template <typename T>
using ff_remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// Dependent false value, used to make static_assert failures depend on a
// template parameter so they only fire when the overload is instantiated.
template <typename...> inline constexpr bool ff_always_false = false;

// Sets a single free function kernel argument. Mirrors handler::setArgHelper
// for the argument kinds supported by the direct submission path.
template <typename T>
void setFreeFunctionArg(FreeFunctionArgCollector &Collector, int ArgIndex,
                        T &&Arg) {
  using BareT = ff_remove_cv_ref_t<T>;
  static_assert(
      !ext::oneapi::experimental::detail::is_struct_with_special_type<
          BareT>::value,
      "Passing a struct that contains SYCL special types (such as accessors) "
      "as a free function kernel argument is not supported on the direct queue "
      "submission path. Submit the kernel through a handler instead, e.g. "
      "nd_launch(handler &, ...).");
  Collector.addPlainArg(kernel_param_kind_t::kind_std_layout, &Arg, sizeof(T),
                        ArgIndex);
}

// Pointer arguments (USM).
template <typename T>
void setFreeFunctionArg(FreeFunctionArgCollector &Collector, int ArgIndex,
                        T *Arg) {
  Collector.addPlainArg(kernel_param_kind_t::kind_pointer, &Arg, sizeof(T *),
                        ArgIndex);
}

// Accessor arguments are not supported on the direct submission path: they
// require memory-object requirement tracking that only the handler provides.
template <typename DataT, int Dims, access::mode AccMode,
          access::target AccTarget, access::placeholder IsPlaceholder>
void setFreeFunctionArg(
    FreeFunctionArgCollector &, int,
    accessor<DataT, Dims, AccMode, AccTarget, IsPlaceholder>) {
  static_assert(
      ff_always_false<DataT>,
      "Passing an accessor as a free function kernel argument is not supported "
      "on the direct queue submission path. Submit the kernel through a "
      "handler instead, e.g. nd_launch(handler &, ...).");
}

template <typename DataT, int Dims>
void setFreeFunctionArg(FreeFunctionArgCollector &, int,
                        local_accessor<DataT, Dims>) {
  static_assert(
      ff_always_false<DataT>,
      "Passing a local_accessor as a free function kernel argument is not "
      "supported on the direct queue submission path. Submit the kernel "
      "through a handler instead, e.g. nd_launch(handler &, ...).");
}

// work_group_memory arguments. The work_group_memory_impl base is sliced out
// (as the handler does in set_arg) and handed to the collector, which copies it
// source-side; buffer_size is private and only read by the runtime.
template <typename DataT, typename PropertyListT>
void setFreeFunctionArg(
    FreeFunctionArgCollector &Collector, int ArgIndex,
    ext::oneapi::experimental::work_group_memory<DataT, PropertyListT> &Arg) {
  work_group_memory_impl &ArgImpl = Arg;
  Collector.addWorkGroupMemoryArg(&ArgImpl, ArgIndex);
}

// Sampler arguments.
inline void setFreeFunctionArg(FreeFunctionArgCollector &Collector,
                               int ArgIndex, const sampler &Arg) {
  Collector.addSamplerArg(&Arg, ArgIndex);
}

// raw_kernel_arg arguments.
inline void
setFreeFunctionArg(FreeFunctionArgCollector &Collector, int ArgIndex,
                   ext::oneapi::experimental::raw_kernel_arg &&Arg) {
  Collector.addRawArg(Arg.MArgData, Arg.MArgSize, ArgIndex);
}

// Collects all explicit arguments of a free function kernel into a
// FreeFunctionArgCollector, assigning consecutive argument indices.
template <typename... ArgsT>
FreeFunctionArgCollector collectFreeFunctionArgs(ArgsT &&...Args) {
  FreeFunctionArgCollector Collector;
  int ArgIndex = 0;
  (void)std::initializer_list<int>{
      (setFreeFunctionArg(Collector, ArgIndex++, std::forward<ArgsT>(Args)),
       0)...};
  return Collector;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
