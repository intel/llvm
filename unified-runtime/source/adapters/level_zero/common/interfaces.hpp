//===--------- interfaces.hpp - Level Zero Adapter -----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <type_traits>
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>
#include <ur/ur.hpp>
#include <vector>
#include <ze_api.h>

namespace ur::level_zero {

struct ur_platform_handle_t_;
typedef struct ur_platform_handle_t_ *ur_platform_handle_t;
struct ur_device_handle_t_;
typedef struct ur_device_handle_t_ *ur_device_handle_t;
struct ur_adapter_handle_t_;
typedef struct ur_adapter_handle_t_ *ur_adapter_handle_t;
struct ur_sampler_handle_t_;
typedef struct ur_sampler_handle_t_ *ur_sampler_handle_t;
struct ur_physical_mem_handle_t_;
typedef struct ur_physical_mem_handle_t_ *ur_physical_mem_handle_t;
struct ur_exp_command_buffer_command_handle_t_;
typedef struct ur_exp_command_buffer_command_handle_t_
    *ur_exp_command_buffer_command_handle_t;
struct ur_program_handle_t_;
typedef struct ur_program_handle_t_ *ur_program_handle_t;

namespace detail {
// Maps an opaque handle typedef to its corresponding internal struct.
template <typename Opaque> struct common_handle_traits;
template <> struct common_handle_traits<::ur_platform_handle_t> {
  using type = ur_platform_handle_t_;
};
template <> struct common_handle_traits<::ur_device_handle_t> {
  using type = ur_device_handle_t_;
};
template <> struct common_handle_traits<::ur_adapter_handle_t> {
  using type = ur_adapter_handle_t_;
};
template <> struct common_handle_traits<::ur_sampler_handle_t> {
  using type = ur_sampler_handle_t_;
};
template <> struct common_handle_traits<::ur_physical_mem_handle_t> {
  using type = ur_physical_mem_handle_t_;
};
template <>
struct common_handle_traits<::ur_exp_command_buffer_command_handle_t> {
  using type = ur_exp_command_buffer_command_handle_t_;
};
template <> struct common_handle_traits<::ur_program_handle_t> {
  using type = ur_program_handle_t_;
};

template <typename Opaque>
using common_internal_t = typename common_handle_traits<Opaque>::type;

// Reverse mapping: internal struct -> opaque handle.
template <typename Internal> struct opaque_handle_for;
template <> struct opaque_handle_for<ur_platform_handle_t_> {
  using type = ::ur_platform_handle_t;
};
template <> struct opaque_handle_for<ur_device_handle_t_> {
  using type = ::ur_device_handle_t;
};
template <> struct opaque_handle_for<ur_adapter_handle_t_> {
  using type = ::ur_adapter_handle_t;
};
template <> struct opaque_handle_for<ur_sampler_handle_t_> {
  using type = ::ur_sampler_handle_t;
};
template <> struct opaque_handle_for<ur_physical_mem_handle_t_> {
  using type = ::ur_physical_mem_handle_t;
};
template <> struct opaque_handle_for<ur_exp_command_buffer_command_handle_t_> {
  using type = ::ur_exp_command_buffer_command_handle_t;
};
template <> struct opaque_handle_for<ur_program_handle_t_> {
  using type = ::ur_program_handle_t;
};
} // namespace detail

// Opaque handle -> internal pointer.
template <typename Opaque>
inline detail::common_internal_t<Opaque> *common_cast(Opaque h) {
  return reinterpret_cast<detail::common_internal_t<Opaque> *>(h);
}

// Opaque handle array -> internal pointer array.
template <typename Opaque>
inline detail::common_internal_t<Opaque> **common_cast(Opaque *ph) {
  return reinterpret_cast<detail::common_internal_t<Opaque> **>(ph);
}

// const Opaque handle array -> const internal pointer array.
template <typename Opaque>
inline detail::common_internal_t<Opaque> *const *common_cast(const Opaque *ph) {
  return reinterpret_cast<detail::common_internal_t<Opaque> *const *>(ph);
}

// Internal pointer -> opaque handle (reverse direction).
template <typename Internal>
inline typename detail::opaque_handle_for<Internal>::type
common_cast(Internal *p) {
  return reinterpret_cast<typename detail::opaque_handle_for<Internal>::type>(
      p);
}

// Internal pointer array -> opaque handle array.
template <typename Internal>
inline typename detail::opaque_handle_for<Internal>::type *
common_cast(Internal **p) {
  return reinterpret_cast<typename detail::opaque_handle_for<Internal>::type *>(
      p);
}

// Element-wise cast of a vector of internal pointers to opaque handles.
template <typename Internal,
          typename = std::void_t<typename detail::opaque_handle_for<
              std::remove_pointer_t<Internal>>::type>>
inline std::vector<
    typename detail::opaque_handle_for<std::remove_pointer_t<Internal>>::type>
common_cast(const std::vector<Internal> &v) {
  using Opaque =
      typename detail::opaque_handle_for<std::remove_pointer_t<Internal>>::type;
  std::vector<Opaque> out;
  out.reserve(v.size());
  for (auto p : v)
    out.push_back(common_cast(p));
  return out;
}

// Head layout shared by every UR handle.
struct handle_head_t {
  const ur_dditable_t *ddi_table;
  ur_shared_mutex Mutex;
};

template <typename H> inline const ur_dditable_t *ddiTableOf(H handle) {
  return reinterpret_cast<const handle_head_t *>(handle)->ddi_table;
}

// getddi policy for handles shared across L0v1 and L0v2, whose `ddi_table` is
// origin-dependent.
struct null_ddi {
  static const ur_dditable_t *value() { return nullptr; }
};

// Primary base for the common concrete handle types. Uses `null_ddi` because
// the table is filled at construction.
struct ur_object_t : ur::handle_base<null_ddi> {
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;
};

// Version-agnostic context base, shared by the L0v1 and L0v2 concrete contexts.
struct ur_context_common_t : ur_object_t {
  ze_context_handle_t ZeContext = nullptr;
  std::vector<ur_device_handle_t> hDevices;
  ur_platform_handle_t Platform = nullptr; // == hDevices[0]->Platform
  ur_shared_mutex *MutexPtr = nullptr;     // == &ur_object_t::Mutex

  ze_context_handle_t getZeHandle() const { return ZeContext; }
  ur_platform_handle_t getPlatform() const { return Platform; }
  const std::vector<ur_device_handle_t> &getDevices() const { return hDevices; }
  ur_shared_mutex &getMutex() const { return *MutexPtr; }

  bool isValidDevice(ur_device_handle_t hDevice) const;

protected:
  ur_context_common_t() = default;
  ur_context_common_t(ze_context_handle_t ZeContext,
                      std::vector<ur_device_handle_t> Devices,
                      ur_platform_handle_t Platform)
      : ZeContext(ZeContext), hDevices(std::move(Devices)), Platform(Platform) {
    MutexPtr = &Mutex;
    if (!hDevices.empty())
      ddi_table = ddiTableOf(hDevices[0]);
  }

  ur_context_common_t(const ur_context_common_t &) = delete;
  ur_context_common_t &operator=(const ur_context_common_t &) = delete;
};

// Bridges the opaque context handle to the shared context base.
inline ur_context_common_t *common_cast(::ur_context_handle_t h) {
  return reinterpret_cast<ur_context_common_t *>(h);
}

} // namespace ur::level_zero
