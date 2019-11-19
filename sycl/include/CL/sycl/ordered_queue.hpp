//==----------------- ordered queue.hpp - SYCL queue -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>

#include <memory>
#include <utility>

namespace cl {
namespace sycl {

// Forward declaration
class context;
class device;
class ordered_queue {
public:
  explicit ordered_queue(const property_list &propList = {})
      : ordered_queue(default_selector(), async_handler{}, propList) {}

  ordered_queue(const async_handler &asyncHandler, const property_list &propList = {})
      : ordered_queue(default_selector(), asyncHandler, propList) {}

  ordered_queue(const device_selector &deviceSelector,
        const property_list &propList = {})
      : ordered_queue(deviceSelector.select_device(), async_handler{}, propList) {}

  ordered_queue(const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {})
      : ordered_queue(deviceSelector.select_device(), asyncHandler, propList) {}

  ordered_queue(const device &syclDevice, const property_list &propList = {})
      : ordered_queue(syclDevice, async_handler{}, propList) {}

  ordered_queue(const device &syclDevice, const async_handler &asyncHandler,
        const property_list &propList = {});

  ordered_queue(const context &syclContext, const device_selector &deviceSelector,
        const property_list &propList = {})
      : ordered_queue(syclContext, deviceSelector,
              detail::getSyclObjImpl(syclContext)->get_async_handler(),
              propList) {}

  ordered_queue(const context &syclContext, const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {});

  ordered_queue(cl_command_queue cl_Queue, const context &syclContext,
        const async_handler &asyncHandler = {});

  ordered_queue(const ordered_queue &rhs) = default;

  ordered_queue(ordered_queue &&rhs) = default;

  ordered_queue &operator=(const ordered_queue &rhs) = default;

  ordered_queue &operator=(ordered_queue &&rhs) = default;

  bool operator==(const ordered_queue &rhs) const { return impl == rhs.impl; }

  bool operator!=(const ordered_queue &rhs) const { return !(*this == rhs); }

  cl_command_queue get() const { return impl->get(); }

  context get_context() const { return impl->get_context(); }

  device get_device() const { return impl->get_device(); }

  bool is_host() const { return impl->is_host(); }

  template <info::ordered_queue param>
  typename info::param_traits<info::ordered_queue, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  template <typename T> event submit(T cgf) { return impl->submit(cgf, impl); }

  template <typename T> event submit(T cgf, ordered_queue &secondaryQueue) {
    return impl->submit(cgf, impl, secondaryQueue.impl);
  }

  void wait() { impl->wait(); }

  void wait_and_throw() { impl->wait_and_throw(); }

  void throw_asynchronous() { impl->throw_asynchronous(); }

  template <typename propertyT> bool has_property() const {
    return impl->has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->get_property<propertyT>();
  }

  event memset(void* ptr, int value, size_t count) {
    return impl->memset(impl, ptr, value, count);
  }

  event memcpy(void* dest, const void* src, size_t count) {
    return impl->memcpy(impl, dest, src, count);
  }

  event prefetch(const void* Ptr, size_t Count) {
    return submit([=](handler &cgh) {
        cgh.prefetch(Ptr, Count);
    });
  }

  // single_task version with a kernel represented as a lambda.
  template <typename KernelName = csd::auto_name, typename KernelType>
  void single_task(KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template single_task<KernelName, KernelType>(KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + range that
  // specifies global size only.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
    // By-value or By-reference for this?
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                              KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(
          NumWorkItems, WorkItemOffset, KernelFunc);
    });
  }

  // parallel_for version with a kernel represented as a lambda + nd_range that
  // specifies global, local sizes and offset.
  template <typename KernelName = csd::auto_name, typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    submit([&](handler &cgh) {
      cgh.template parallel_for<KernelName, KernelType, Dims>(ExecutionRange,
                                                              KernelFunc);
    });
  }

private:
  std::shared_ptr<detail::queue_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
};

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
