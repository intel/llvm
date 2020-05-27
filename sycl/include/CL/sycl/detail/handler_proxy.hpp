#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/export.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class handler;

namespace detail {

class AccessorBaseHost;

__SYCL_EXPORT void associateWithHandler(handler &, AccessorBaseHost *,
                                        access::target);
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
