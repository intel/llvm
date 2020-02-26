//==-- interop_handler.hpp --- SYCL interoperability handler ---*- C++ -*---==//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive property
// of Intel Corporation and may not be disclosed, examined or reproduced in
// whole or in part without explicit written authorization from the company.
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/cl.h>

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;

class interop_handle {
private:
  interop_handle() {
    assert(false && "Not implemented");
  }

public:
  template <typename DataT, int Dimensions, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  cl_mem get_native_mem(const accessor<DataT, Dimensions, AccessMode,
                                       AccessTarget, IsPlaceholder> &) const {
    assert(false && "Not implemented");
  }

  cl_command_queue get_native_queue() const noexcept {
    assert(false && "Not implemented");
  }

  cl_device_id get_native_device() const noexcept {
    assert(false && "Not implemented");
  }

  cl_context get_native_context() const noexcept {
    assert(false && "Not implemented");
  }
};

} // namespace sycl
} // namespace cl
