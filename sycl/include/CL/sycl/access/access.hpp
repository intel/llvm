//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

namespace cl {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class fence_space {
  local_space,
  global_space,
  global_and_local
};

enum class placeholder { false_t, true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space
};

}  // namespace access

namespace detail {

constexpr bool isTargetHostAccess(access::target T) {
  return T == access::target::host_buffer || T == access::target::host_image;
}

constexpr bool modeNeedsOldData(access::mode m) {
  return m == access::mode::read || m == access::mode::write ||
         m == access::mode::read_write || m == access::mode::atomic;
}

constexpr bool modeWritesNewData(access::mode m) {
  return m != access::mode::read;
}

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_GLOBAL_AS __attribute__((ocl_global))
#define SYCL_LOCAL_AS __attribute__((ocl_local))
#define SYCL_CONSTANT_AS __attribute__((ocl_constant))
#define SYCL_PRIVATE_AS __attribute__((ocl_private))
#else
#define SYCL_GLOBAL_AS
#define SYCL_LOCAL_AS
#define SYCL_CONSTANT_AS
#define SYCL_PRIVATE_AS
#endif

template <typename dataT, access::target accessTarget>
struct DeviceValueType;

template <typename dataT>
struct DeviceValueType<dataT, access::target::global_buffer> {
  using type = SYCL_GLOBAL_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::constant_buffer> {
  using type = SYCL_CONSTANT_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::local> {
  using type = SYCL_LOCAL_AS dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::host_buffer> {
  using type = dataT;
};

template <access::target accessTarget> struct TargetToAS {
  constexpr static access::address_space AS =
      access::address_space::global_space;
};

template <> struct TargetToAS<access::target::local> {
  constexpr static access::address_space AS =
      access::address_space::local_space;
};

template <> struct TargetToAS<access::target::constant_buffer> {
  constexpr static access::address_space AS =
      access::address_space::constant_space;
};

template <typename ElementType, access::address_space addressSpace>
struct PtrValueType;

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::private_space> {
  using type = SYCL_PRIVATE_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::global_space> {
  using type = SYCL_GLOBAL_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::constant_space> {
  // Current implementation of address spaces handling leads to possibility
  // of emitting incorrect (in terms of OpenCL) address space casts from
  // constant to generic (and vise-versa). So, global address space is used here
  // instead of constant to avoid incorrect address space casts in the produced
  // device code. "const" qualifier is not used here because multi_ptr interface
  // contains function members which return pure ElementType without qualifiers
  // and adding const qualifier here will require adding const casts to
  // multi_ptr methods to remove const qualifiers from underlying pointer type.
  using type = SYCL_GLOBAL_AS ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::local_space> {
  using type = SYCL_LOCAL_AS ElementType;
};

template <class T>
struct remove_AS {
  typedef T type;
};

#ifdef __SYCL_DEVICE_ONLY__
template <class T>
struct remove_AS<SYCL_GLOBAL_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_PRIVATE_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_LOCAL_AS T> {
  typedef T type;
};

template <class T>
struct remove_AS<SYCL_CONSTANT_AS T> {
  typedef T type;
};
#endif

#undef SYCL_GLOBAL_AS
#undef SYCL_LOCAL_AS
#undef SYCL_CONSTANT_AS
#undef SYCL_PRIVATE_AS

} // namespace detail

}  // namespace sycl
}  // namespace cl
