//==--- raw_kernel_arg.hpp --- SYCL extension for raw kernel args ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stddef.h>

namespace sycl {
inline namespace _V1 {

class handler;
namespace ext::oneapi::experimental {

namespace detail {
class dynamic_parameter_impl;
struct RawKernelArgAccess;
} // namespace detail

// Tells the raw_kernel_arg constructor below that the argument it is given is a
// pointer rather than a sequence of bytes to copy.
struct pointer_arg_t {};
inline constexpr pointer_arg_t pointer_arg{};

class raw_kernel_arg {
public:
  raw_kernel_arg(const void *bytes, size_t count)
      : MArgData(bytes), MArgSize(count) {}

  // A pointer argument is not interchangeable with the bytes it is made of: a
  // backend may have to be told that an argument is a pointer to bind it at
  // all, as OpenCL does, where a USM pointer goes to
  // clSetKernelArgMemPointerINTEL rather than to clSetKernelArg. Takes the
  // address of the pointer, like the byte form takes the address of the bytes,
  // so that passing the pointer itself does not compile.
  template <typename T>
  raw_kernel_arg(T *const *pointer_location, pointer_arg_t)
      : MArgData(pointer_location), MArgSize(sizeof(T *)), MIsPointer(true) {}

private:
  const void *MArgData;
  size_t MArgSize;
  // Appended last, so that the offsets of the members above, which the library
  // reads on the paths that predate this one, stay where they were.
  bool MIsPointer = false;

  friend class sycl::handler;
  // For sycl_ext_oneapi_graph integration
  friend class detail::dynamic_parameter_impl;
  // For the enqueue paths that bind arguments without a handler
  friend struct detail::RawKernelArgAccess;
};

namespace detail {
// Helper for accessing the members of raw_kernel_arg.
struct RawKernelArgAccess {
  static const void *getData(const raw_kernel_arg &Arg) { return Arg.MArgData; }
  static size_t getSize(const raw_kernel_arg &Arg) { return Arg.MArgSize; }
  static bool isPointer(const raw_kernel_arg &Arg) { return Arg.MIsPointer; }
};
} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
