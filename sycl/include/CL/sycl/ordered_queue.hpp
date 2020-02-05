//==----------------- ordered queue.hpp - SYCL queue -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>

#include <memory>
#include <utility>

#ifdef __has_cpp_attribute
  #if __has_cpp_attribute(deprecated)
    #define __SYCL_DEPRECATED__ [[deprecated("Replaced by in_order queue property")]]
  #endif
#endif
#ifndef __SYCL_DEPRECATED__
  #define __SYCL_DEPRECATED__
#endif

__SYCL_INLINE namespace cl {
namespace sycl {

// Forward declaration
class context;
class device;
namespace detail {
class queue_impl;
}

class __SYCL_DEPRECATED__ ordered_queue {

public:
  explicit ordered_queue(const property_list &propList = {})
      : ordered_queue(default_selector(), async_handler{}, propList) {}

  ordered_queue(const async_handler &asyncHandler,
                const property_list &propList = {})
      : ordered_queue(default_selector(), asyncHandler, propList) {}

  ordered_queue(const device_selector &deviceSelector,
                const property_list &propList = {})
      : ordered_queue(deviceSelector.select_device(), async_handler{},
                      propList) {}
  ordered_queue(const device_selector &deviceSelector,
                const async_handler &asyncHandler,
                const property_list &propList = {})
      : ordered_queue(deviceSelector.select_device(), asyncHandler, propList) {}

  ordered_queue(const device &syclDevice, const property_list &propList = {})
      : ordered_queue(syclDevice, async_handler{}, propList) {}

  ordered_queue(const device &syclDevice, const async_handler &asyncHandler,
                const property_list &propList = {});

  ordered_queue(const context &syclContext,
                const device_selector &deviceSelector,
                const property_list &propList = {});

  ordered_queue(const context &syclContext,
                const device_selector &deviceSelector,
                const async_handler &asyncHandler,
                const property_list &propList = {});

  ordered_queue(cl_command_queue cl_Queue, const context &syclContext,
                const async_handler &asyncHandler = {});

  ordered_queue(const ordered_queue &rhs) = default;

  ordered_queue(ordered_queue &&rhs) = default;

  ordered_queue &operator=(const ordered_queue &rhs) = default;

  ordered_queue &operator=(ordered_queue &&rhs) = default;

  bool operator==(const ordered_queue &rhs) const { return impl == rhs.impl; }

  bool operator!=(const ordered_queue &rhs) const { return !(*this == rhs); }

  cl_command_queue get() const;

  context get_context() const;

  device get_device() const;

  bool is_host() const;

  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

  template <typename T> event submit(T cgf) { return submit_impl(cgf); }

  template <typename T> event submit(T cgf, ordered_queue &secondaryQueue) {
    return submit_impl(cgf, secondaryQueue);
  }

  void wait();

  void wait_and_throw();

  void throw_asynchronous();

  template <typename propertyT> bool has_property() const;

  template <typename propertyT> propertyT get_property() const;

  event memset(void *ptr, int value, size_t count);

  event memcpy(void *dest, const void *src, size_t count);

  event prefetch(const void *Ptr, size_t Count) {
    return submit([=](handler &cgh) { cgh.prefetch(Ptr, Count); });
  }

  // single_task version with a kernel represented as a lambda.
  template <typename KernelName = detail::auto_name, typename KernelType>
  void single_task(KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template single_task<KernelName, KernelType>(KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + range that
  // specifies global size only.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
    // By-value or By-reference for this?
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                              KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(
          NumWorkItems, WorkItemOffset, KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + nd_range that
  // specifies global, local sizes and offset.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(ExecutionRange,
                                                              KernelFunc);
    });
  }

private:
  shared_ptr_class<detail::queue_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  event submit_impl(function_class<void(handler &)> CGH);
  event submit_impl(function_class<void(handler &)> CGH,
                    ordered_queue &secondQueue);
};

#undef __SYCL_DEPRECATED__

} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::ordered_queue> {
  size_t operator()(const cl::sycl::ordered_queue &q) const {
    return std::hash<std::shared_ptr<cl::sycl::detail::queue_impl>>()(
        cl::sycl::detail::getSyclObjImpl(q));
  }
};
} // namespace std
