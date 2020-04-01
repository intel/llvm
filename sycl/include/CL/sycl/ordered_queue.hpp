//==----------------- ordered queue.hpp - SYCL queue -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>

#include <memory>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
class context;
class device;
namespace detail {
class queue_impl;
}

class __SYCL_DEPRECATED__("Replaced by in_order queue property") ordered_queue {

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

  /// @param Loc is the code location of the submit call (default argument)
  template <typename T>
  event
  submit(T cgf
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
         ,
         const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit_impl(cgf, CodeLoc);
  }

  template <typename T>
  event
  submit(T cgf, ordered_queue &secondaryQueue
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
         ,
         const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit_impl(cgf, secondaryQueue, CodeLoc);
  }

  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait(
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    wait_proxy(CodeLoc);
  }

  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait_and_throw(
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    wait_and_throw_proxy(CodeLoc);
  }

  void wait_proxy(const detail::code_location &CodeLoc);

  void wait_and_throw_proxy(const detail::code_location &CodeLoc);

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
  void single_task(
      KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    submit(
        [&](handler &cgh) {
          cgh.template single_task<KernelName, KernelType>(KernelFunc);
        },
        CodeLoc);
  }

  // parallel_for version with a kernel represented as a lambda + range that
  // specifies global size only.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(
      range<Dims> NumWorkItems, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    // By-value or By-reference for this?
    submit(
        [&](handler &cgh) {
          cgh.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                                  KernelFunc);
        },
        CodeLoc);
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(
      range<Dims> NumWorkItems, id<Dims> WorkItemOffset, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    submit(
        [&](handler &cgh) {
          cgh.template parallel_for<KernelName, KernelType, Dims>(
              NumWorkItems, WorkItemOffset, KernelFunc);
        },
        CodeLoc);
  }

  // parallel_for version with a kernel represented as a lambda + nd_range that
  // specifies global, local sizes and offset.
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(
      nd_range<Dims> ExecutionRange, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    submit(
        [&](handler &cgh) {
          cgh.template parallel_for<KernelName, KernelType, Dims>(
              ExecutionRange, KernelFunc);
        },
        CodeLoc);
  }

private:
  shared_ptr_class<detail::queue_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  event submit_impl(function_class<void(handler &)> CGH,
                    const detail::code_location &CodeLoc);
  event submit_impl(function_class<void(handler &)> CGH,
                    ordered_queue &secondQueue,
                    const detail::code_location &CodeLoc);
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::ordered_queue> {
  size_t operator()(const cl::sycl::ordered_queue &q) const {
    return std::hash<std::shared_ptr<cl::sycl::detail::queue_impl>>()(
        cl::sycl::detail::getSyclObjImpl(q));
  }
};
} // namespace std
