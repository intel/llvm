//===--------- context_interface.hpp - Level Zero Adapter -----------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>
#include <vector>
#include <ze_api.h>

// Dispatch surface that common code depends on for `ur_context_handle_t_`.
// v1 (`level_zero/context.hpp`) and v2 (`v2/context.hpp`) each define their
// own concrete `ur_context_handle_t_` that inherits `ur_context_interface_t`
// as its FIRST base class.
//
// The interface is DELIBERATELY non-virtual. A virtual class would force the
// compiler to place a vtable pointer at offset 0 of every derived object,
// displacing `ddi_table` — but the loader's intercept layer in
// `source/loader/ur_ldrddi.cpp` reads `*reinterpret_cast<ur_dditable_t **>(h)`
// at offset 0 of every opaque handle. Dispatch is therefore done through a
// per-adapter function-pointer table (`ur_context_vfns_t`) that the
// adapter's concrete ctor registers with the interface.
struct ur_context_interface_t;

// Function-pointer dispatch table. Each adapter populates a `static constexpr`
// instance of this and passes it to the interface base ctor.
struct ur_context_vfns_t {
  ze_context_handle_t (*getZeHandle)(const ur_context_interface_t *);
  ur_platform_handle_t (*getPlatform)(const ur_context_interface_t *);
  const std::vector<ur_device_handle_t> &(*getDevices)(
      const ur_context_interface_t *);
  bool (*isValidDevice)(const ur_context_interface_t *, ur_device_handle_t);
  ur_shared_mutex &(*getMutex)(ur_context_interface_t *);
};

struct ur_context_interface_t {
  // Loader intercept reads this at offset 0 of the handle.
  const ur_dditable_t *ddi_table;
  // Adapter-provided dispatch table for the methods below.
  const ur_context_vfns_t *vfns;

  // Inline wrappers that forward through the vfns table. Existing
  // `asInterface(h)->getDevices()`-style call sites in common/ keep
  // working unchanged.
  ze_context_handle_t getZeHandle() const { return vfns->getZeHandle(this); }
  ur_platform_handle_t getPlatform() const { return vfns->getPlatform(this); }
  const std::vector<ur_device_handle_t> &getDevices() const {
    return vfns->getDevices(this);
  }
  bool isValidDevice(ur_device_handle_t hDevice) const {
    return vfns->isValidDevice(this, hDevice);
  }
  ur_shared_mutex &getMutex() { return vfns->getMutex(this); }

protected:
  ur_context_interface_t(const ur_dditable_t *ddi,
                         const ur_context_vfns_t *fns)
      : ddi_table(ddi), vfns(fns) {}
  ur_context_interface_t(const ur_context_interface_t &) = delete;
  ur_context_interface_t &operator=(const ur_context_interface_t &) = delete;
};

// Reinterprets the opaque UR handle as a pointer to the interface. Safe
// because both adapters' concrete `ur_context_handle_t_` inherit
// `ur_context_interface_t` as their first base — its subobject sits at
// offset 0 of the concrete handle.
inline ur_context_interface_t *asInterface(ur_context_handle_t h) {
  return reinterpret_cast<ur_context_interface_t *>(h);
}

// Reads `ddi_table` at offset 0 of an opaque UR handle. Every UR handle
// carries `ddi_table` as its first member (the loader's intercept layer
// depends on this). This helper lets common code source an adapter's DDI
// pointer from any handle without needing the full struct definition.
template <typename H>
inline const ur_dditable_t *ddiTableOf(H handle) {
  struct ddi_head_t {
    const ur_dditable_t *ddi_table;
  };
  return reinterpret_cast<const ddi_head_t *>(handle)->ddi_table;
}
