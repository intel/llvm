//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/kernel_info.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/queue.hpp>

#include <cassert>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {
// Forward declaration
class kernel_bundle_impl;

using ContextImplPtr = std::shared_ptr<context_impl>;
using KernelBundleImplPtr = std::shared_ptr<kernel_bundle_impl>;
class kernel_impl {
public:
  /// Constructs a SYCL kernel instance from a UrKernel
  ///
  /// This constructor is used for UR adapter interoperability. It always marks
  /// kernel as being created from source.
  ///
  /// \param Kernel is a valid UrKernel instance
  /// \param Context is a valid SYCL context
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(ur_kernel_handle_t Kernel, ContextImplPtr Context,
              KernelBundleImplPtr KernelBundleImpl,
              const KernelArgMask *ArgMask = nullptr);

  /// Constructs a SYCL kernel_impl instance from a SYCL device_image,
  /// kernel_bundle and / UrKernel.
  ///
  /// \param Kernel is a valid UrKernel instance
  /// \param ContextImpl is a valid SYCL context
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(ur_kernel_handle_t Kernel, ContextImplPtr ContextImpl,
              DeviceImageImplPtr DeviceImageImpl,
              KernelBundleImplPtr KernelBundleImpl,
              const KernelArgMask *ArgMask, ur_program_handle_t Program,
              std::mutex *CacheMutex);

  // This section means the object is non-movable and non-copyable
  // There is no need of move and copy constructors in kernel_impl.
  // If they need to be added, urKernelRetain method for MKernel
  // should be present.
  kernel_impl(const kernel_impl &) = delete;
  kernel_impl(kernel_impl &&) = delete;
  kernel_impl &operator=(const kernel_impl &) = delete;
  kernel_impl &operator=(kernel_impl &&) = delete;

  ~kernel_impl();

  /// Gets a valid OpenCL kernel handle
  ///
  /// If this kernel encapsulates an instance of OpenCL kernel, a valid
  /// cl_kernel will be returned. If this kernel is a host kernel,
  /// an exception with errc::invalid error code will be thrown.
  ///
  /// \return a valid cl_kernel instance
  cl_kernel get() const {
    getAdapter()->call<UrApiKind::urKernelRetain>(MKernel);
    ur_native_handle_t nativeHandle = 0;
    getAdapter()->call<UrApiKind::urKernelGetNativeHandle>(MKernel,
                                                           &nativeHandle);
    return ur::cast<cl_kernel>(nativeHandle);
  }

  const AdapterPtr &getAdapter() const { return MContext->getAdapter(); }

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// \return depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries the kernel object for SYCL backend-specific information.
  ///
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Query device-specific information from a kernel object using the
  /// info::kernel_device_specific descriptor.
  ///
  /// \param Device is a valid SYCL device to query info for.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_info(const device &Device) const;

  /// Query device-specific information from a kernel using the
  /// info::kernel_device_specific descriptor for a specific device and value.
  /// max_sub_group_size is the only valid descriptor for this function.
  ///
  /// \param Device is a valid SYCL device.
  /// \param WGSize is the work-group size the sub-group size is requested for.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_info(const device &Device,
                                       const range<3> &WGSize) const;

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type ext_oneapi_get_info(queue Queue) const;

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue and values.
  /// max_num_work_groups is the only valid descriptor for this function.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \param WorkGroupSize is the work-group size the number of work-groups is
  /// requested for.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type
  ext_oneapi_get_info(queue Queue, const range<3> &MaxWorkGroupSize,
                      size_t DynamicLocalMemorySize) const;

  /// Get a constant reference to a raw kernel object.
  ///
  /// \return a constant reference to a valid UrKernel instance with raw
  /// kernel object.
  const ur_kernel_handle_t &getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// \return true if kernel was created from source.
  bool isCreatedFromSource() const;

  const DeviceImageImplPtr &getDeviceImage() const { return MDeviceImageImpl; }

  ur_native_handle_t getNative() const {
    const AdapterPtr &Adapter = MContext->getAdapter();

    if (MContext->getBackend() == backend::opencl)
      Adapter->call<UrApiKind::urKernelRetain>(MKernel);

    ur_native_handle_t NativeKernel = 0;
    Adapter->call<UrApiKind::urKernelGetNativeHandle>(MKernel, &NativeKernel);

    return NativeKernel;
  }

  KernelBundleImplPtr get_kernel_bundle() const { return MKernelBundleImpl; }

  bool isInterop() const { return MIsInterop; }

  ur_program_handle_t getProgramRef() const { return MProgram; }
  ContextImplPtr getContextImplPtr() const { return MContext; }

  std::mutex &getNoncacheableEnqueueMutex() {
    return MNoncacheableEnqueueMutex;
  }

  const KernelArgMask *getKernelArgMask() const { return MKernelArgMaskPtr; }
  std::mutex *getCacheMutex() const { return MCacheMutex; }

private:
  ur_kernel_handle_t MKernel = nullptr;
  const ContextImplPtr MContext;
  const ur_program_handle_t MProgram = nullptr;
  bool MCreatedFromSource = true;
  const DeviceImageImplPtr MDeviceImageImpl;
  const KernelBundleImplPtr MKernelBundleImpl;
  bool MIsInterop = false;
  std::mutex MNoncacheableEnqueueMutex;
  const KernelArgMask *MKernelArgMaskPtr;
  std::mutex *MCacheMutex = nullptr;

  bool isBuiltInKernel(const device &Device) const;
  void checkIfValidForNumArgsInfoQuery() const;

  /// Check if the occupancy limits are exceeded for the given kernel launch
  /// configuration.
  bool exceedsOccupancyResourceLimits(const device &Device,
                                      const range<3> &WorkGroupSize,
                                      size_t DynamicLocalMemorySize) const;
};

template <typename Param>
inline typename Param::return_type kernel_impl::get_info() const {
  static_assert(is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  if constexpr (std::is_same_v<Param, info::kernel::num_args>)
    checkIfValidForNumArgsInfoQuery();

  return get_kernel_info<Param>(this->getHandleRef(), getAdapter());
}

template <>
inline context kernel_impl::get_info<info::kernel::context>() const {
  return createSyclObjFromImpl<context>(MContext);
}

template <typename Param>
inline typename Param::return_type
kernel_impl::get_info(const device &Device) const {
  if constexpr (std::is_same_v<
                    Param, info::kernel_device_specific::global_work_size>) {
    bool isDeviceCustom = Device.get_info<info::device::device_type>() ==
                          info::device_type::custom;
    if (!isDeviceCustom && !isBuiltInKernel(Device))
      throw exception(
          sycl::make_error_code(errc::invalid),
          "info::kernel_device_specific::global_work_size descriptor may only "
          "be used if the device type is device_type::custom or if the kernel "
          "is a built-in kernel.");
  }

  return get_kernel_device_specific_info<Param>(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(),
      getAdapter());
}

template <typename Param>
inline typename Param::return_type
kernel_impl::get_info(const device &Device,
                      const sycl::range<3> &WGSize) const {
  return get_kernel_device_specific_info_with_input<Param>(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(), WGSize,
      getAdapter());
}

namespace syclex = ext::oneapi::experimental;

template <>
inline typename syclex::info::kernel_queue_specific::max_num_work_groups::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(
        queue Queue, const range<3> &WorkGroupSize,
        size_t DynamicLocalMemorySize) const {
  if (WorkGroupSize.size() == 0)
    throw exception(sycl::make_error_code(errc::invalid),
                    "The launch work-group size cannot be zero.");

  const auto &Adapter = getAdapter();
  const auto &Handle = getHandleRef();
  auto Device = Queue.get_device();

  uint32_t GroupCount{0};
  if (auto Result = Adapter->call_nocheck<
                    UrApiKind::urKernelSuggestMaxCooperativeGroupCountExp>(
          Handle, WorkGroupSize.size(), DynamicLocalMemorySize, &GroupCount);
      Result != UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    // The feature is supported. Check for other errors and throw if any.
    Adapter->checkUrResult(Result);
    return GroupCount;
  }

  // Fallback. If the backend API is unsupported, this query will return either
  // 0 or 1 based on the kernel resource usage and the user-requested resources.
  return exceedsOccupancyResourceLimits(Device, WorkGroupSize,
                                        DynamicLocalMemorySize)
             ? 0
             : 1;
}

template <>
inline typename syclex::info::kernel_queue_specific::max_num_work_group_sync::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_group_sync>(
        queue Queue, const range<3> &WorkGroupSize,
        size_t DynamicLocalMemorySize) const {
  return ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      Queue, WorkGroupSize, DynamicLocalMemorySize);
}

template <>
inline typename syclex::info::kernel_queue_specific::max_num_work_group_sync::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_group_sync>(
        queue Queue) const {
  auto Device = Queue.get_device();
  const auto MaxWorkGroupSize =
      get_info<info::kernel_device_specific::work_group_size>(Device);
  const sycl::range<3> WorkGroupSize{MaxWorkGroupSize, 1, 1};
  return ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_group_sync>(
      Queue, WorkGroupSize, /* DynamicLocalMemorySize */ 0);
}

template <>
inline typename syclex::info::kernel_queue_specific::max_work_group_size::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_work_group_size>(
        queue Queue) const {
  auto Device = Queue.get_device();
  return get_info<info::kernel_device_specific::work_group_size>(Device);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
