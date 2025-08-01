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
  kernel_impl(Managed<ur_kernel_handle_t> &&Kernel, context_impl &Context,
              kernel_bundle_impl *KernelBundleImpl,
              const KernelArgMask *ArgMask = nullptr);

  /// Constructs a SYCL kernel_impl instance from a SYCL device_image,
  /// kernel_bundle and / UrKernel.
  ///
  /// \param Kernel is a valid UrKernel instance
  /// \param ContextImpl is a valid SYCL context
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(Managed<ur_kernel_handle_t> &&Kernel, context_impl &ContextImpl,
              std::shared_ptr<device_image_impl> &&DeviceImageImpl,
              const kernel_bundle_impl &KernelBundleImpl,
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
    ur_native_handle_t nativeHandle = 0;
    getAdapter().call<UrApiKind::urKernelGetNativeHandle>(MKernel,
                                                          &nativeHandle);
    __SYCL_OCL_CALL(clRetainKernel, ur::cast<cl_kernel>(nativeHandle));
    return ur::cast<cl_kernel>(nativeHandle);
  }

  adapter_impl &getAdapter() const { return MContext->getAdapter(); }

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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // This function is unused and should be removed in the next ABI breaking.

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type ext_oneapi_get_info(queue Queue) const;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

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
  ext_oneapi_get_info(queue Queue, const range<1> &MaxWorkGroupSize,
                      size_t DynamicLocalMemorySize) const;

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
  ext_oneapi_get_info(queue Queue, const range<2> &MaxWorkGroupSize,
                      size_t DynamicLocalMemorySize) const;

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

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue and values.
  /// max_num_work_groups is the only valid descriptor for this function.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \param WG is a work group size
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type ext_oneapi_get_info(queue Queue,
                                                  const range<3> &WG) const;

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue and values.
  /// max_num_work_groups is the only valid descriptor for this function.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \param WG is a work group size
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type ext_oneapi_get_info(queue Queue,
                                                  const range<2> &WG) const;

  /// Query queue/launch-specific information from a kernel using the
  /// info::kernel_queue_specific descriptor for a specific Queue and values.
  /// max_num_work_groups is the only valid descriptor for this function.
  ///
  /// \param Queue is a valid SYCL queue.
  /// \param WG is a work group size
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type ext_oneapi_get_info(queue Queue,
                                                  const range<1> &WG) const;

  ur_kernel_handle_t getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// \return true if kernel was created from source.
  bool isCreatedFromSource() const;

  bool isInteropOrSourceBased() const noexcept;
  bool hasSYCLMetadata() const noexcept;

  device_image_impl &getDeviceImage() const { return *MDeviceImageImpl; }

  ur_native_handle_t getNative() const {
    adapter_impl &Adapter = MContext->getAdapter();

    ur_native_handle_t NativeKernel = 0;
    Adapter.call<UrApiKind::urKernelGetNativeHandle>(MKernel, &NativeKernel);

    if (MContext->getBackend() == backend::opencl)
      __SYCL_OCL_CALL(clRetainKernel, ur::cast<cl_kernel>(NativeKernel));

    return NativeKernel;
  }

  KernelBundleImplPtr get_kernel_bundle() const { return MKernelBundleImpl; }

  bool isInterop() const { return MIsInterop; }

  ur_program_handle_t getProgramRef() const { return MProgram; }
  context_impl &getContextImpl() const { return *MContext; }

  std::mutex &getNoncacheableEnqueueMutex() const {
    return MNoncacheableEnqueueMutex;
  }

  const KernelArgMask *getKernelArgMask() const { return MKernelArgMaskPtr; }
  std::mutex *getCacheMutex() const { return MCacheMutex; }
  std::string_view getName() const;

private:
  Managed<ur_kernel_handle_t> MKernel;
  const std::shared_ptr<context_impl> MContext;
  const ur_program_handle_t MProgram = nullptr;
  bool MCreatedFromSource = true;
  const std::shared_ptr<device_image_impl> MDeviceImageImpl;
  const KernelBundleImplPtr MKernelBundleImpl;
  bool MIsInterop = false;
  mutable std::mutex MNoncacheableEnqueueMutex;
  const KernelArgMask *MKernelArgMaskPtr;
  std::mutex *MCacheMutex = nullptr;
  mutable std::string MName;

  bool isBuiltInKernel(device_impl &Device) const;
  void checkIfValidForNumArgsInfoQuery() const;

  /// Check if the occupancy limits are exceeded for the given kernel launch
  /// configuration.
  template <int Dimensions>
  bool exceedsOccupancyResourceLimits(const device &Device,
                                      const range<Dimensions> &WorkGroupSize,
                                      size_t DynamicLocalMemorySize) const;
  template <int Dimensions>
  size_t queryMaxNumWorkGroups(queue Queue,
                               const range<Dimensions> &WorkGroupSize,
                               size_t DynamicLocalMemorySize) const;

  void enableUSMIndirectAccess() const;
  std::optional<unsigned> getFreeFuncKernelArgSize() const;
};

template <int Dimensions>
bool kernel_impl::exceedsOccupancyResourceLimits(
    const device &Device, const range<Dimensions> &WorkGroupSize,
    size_t DynamicLocalMemorySize) const {
  // Respect occupancy limits for WorkGroupSize and DynamicLocalMemorySize.
  // Generally, exceeding hardware resource limits will yield in an error when
  // the kernel is launched.
  const size_t MaxWorkGroupSize =
      get_info<info::kernel_device_specific::work_group_size>(Device);
  const size_t MaxLocalMemorySizeInBytes =
      Device.get_info<info::device::local_mem_size>();

  if (WorkGroupSize.size() > MaxWorkGroupSize)
    return true;

  if (DynamicLocalMemorySize > MaxLocalMemorySizeInBytes)
    return true;

  // It will be impossible to launch a kernel for Cuda when the hardware limit
  // for the 32-bit registers page file size is exceeded.
  if (Device.get_backend() == backend::ext_oneapi_cuda) {
    const uint32_t RegsPerWorkItem =
        get_info<info::kernel_device_specific::ext_codeplay_num_regs>(Device);
    const uint32_t MaxRegsPerWorkGroup =
        Device.get_info<ext::codeplay::experimental::info::device::
                            max_registers_per_work_group>();
    if ((MaxWorkGroupSize * RegsPerWorkItem) > MaxRegsPerWorkGroup)
      return true;
  }

  return false;
}

template <typename Param>
inline typename Param::return_type kernel_impl::get_info() const {
  static_assert(is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  if constexpr (std::is_same_v<Param, info::kernel::num_args>) {
    // if kernel is a free function, we need to get num_args from integration
    // header, stored in program manager
    if (std::optional<unsigned> FFArgSize = getFreeFuncKernelArgSize())
      return *FFArgSize;
    checkIfValidForNumArgsInfoQuery();
  }
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
    if (!isDeviceCustom && !isBuiltInKernel(*getSyclObjImpl(Device)))
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

template <int Dimensions>
size_t
kernel_impl::queryMaxNumWorkGroups(queue Queue,
                                   const range<Dimensions> &WorkGroupSize,
                                   size_t DynamicLocalMemorySize) const {
  if (WorkGroupSize.size() == 0)
    throw exception(sycl::make_error_code(errc::invalid),
                    "The launch work-group size cannot be zero.");

  adapter_impl &Adapter = getAdapter();
  const auto &Handle = getHandleRef();
  auto Device = Queue.get_device();
  auto DeviceHandleRef = sycl::detail::getSyclObjImpl(Device)->getHandleRef();

  size_t WG[Dimensions];
  WG[0] = WorkGroupSize[0];
  if constexpr (Dimensions >= 2)
    WG[1] = WorkGroupSize[1];
  if constexpr (Dimensions == 3)
    WG[2] = WorkGroupSize[2];

  uint32_t GroupCount{0};
  if (auto Result =
          Adapter
              .call_nocheck<UrApiKind::urKernelSuggestMaxCooperativeGroupCount>(
                  Handle, DeviceHandleRef, Dimensions, WG,
                  DynamicLocalMemorySize, &GroupCount);
      Result != UR_RESULT_ERROR_UNSUPPORTED_FEATURE &&
      Result != UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE) {
    // The feature is supported and the group size is valid. Check for other
    // errors and throw if any.
    Adapter.checkUrResult(Result);
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
inline typename syclex::info::kernel_queue_specific::max_num_work_groups::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(
        queue Queue, const range<1> &WorkGroupSize,
        size_t DynamicLocalMemorySize) const {
  return queryMaxNumWorkGroups(std::move(Queue), WorkGroupSize,
                               DynamicLocalMemorySize);
}

template <>
inline typename syclex::info::kernel_queue_specific::max_num_work_groups::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(
        queue Queue, const range<2> &WorkGroupSize,
        size_t DynamicLocalMemorySize) const {
  return queryMaxNumWorkGroups(std::move(Queue), WorkGroupSize,
                               DynamicLocalMemorySize);
}

template <>
inline typename syclex::info::kernel_queue_specific::max_num_work_groups::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(
        queue Queue, const range<3> &WorkGroupSize,
        size_t DynamicLocalMemorySize) const {
  return queryMaxNumWorkGroups(std::move(Queue), WorkGroupSize,
                               DynamicLocalMemorySize);
}

template <>
inline typename ext::intel::info::kernel_device_specific::spill_memory_size::
    return_type
    kernel_impl::get_info<
        ext::intel::info::kernel_device_specific::spill_memory_size>(
        const device &Device) const {
  if (!Device.has(aspect::ext_intel_spill_memory_size))
    throw exception(
        make_error_code(errc::feature_not_supported),
        "This device does not have the ext_intel_spill_memory_size aspect");

  return get_kernel_device_specific_info<
      ext::intel::info::kernel_device_specific::spill_memory_size>(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(),
      getAdapter());
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// These functions are unused and should be removed in the next ABI breaking.

template <>
inline typename syclex::info::kernel_queue_specific::max_work_group_size::
    return_type
    kernel_impl::ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_work_group_size>(
        queue Queue) const {
  adapter_impl &Adapter = getAdapter();
  const auto DeviceNativeHandle =
      getSyclObjImpl(Queue.get_device())->getHandleRef();

  size_t KernelWGSize = 0;
  Adapter.call<UrApiKind::urKernelGetGroupInfo>(
      MKernel, DeviceNativeHandle, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
      sizeof(size_t), &KernelWGSize, nullptr);
  return KernelWGSize;
}

template <int Dimensions>
inline sycl::id<Dimensions>
generate_id(const sycl::range<Dimensions> &DevMaxWorkItemSizes,
            const size_t DevWgSize) {
  sycl::id<Dimensions> Ret;
  for (int i = 0; i < Dimensions; i++) {
    // DevMaxWorkItemSizes values are inverted, see
    // sycl/source/detail/device_info.hpp:582
    Ret[i] = std::min(DevMaxWorkItemSizes[i], DevWgSize);
  }
  return Ret;
}

#define ADD_TEMPLATE_METHOD_SPEC(Num)                                          \
  template <>                                                                  \
  inline typename syclex::info::kernel_queue_specific::max_work_item_sizes<    \
      Num>::return_type                                                        \
  kernel_impl::ext_oneapi_get_info<                                            \
      syclex::info::kernel_queue_specific::max_work_item_sizes<Num>>(          \
      queue Queue) const {                                                     \
    const auto Dev = Queue.get_device();                                       \
    const auto DeviceWgSize =                                                  \
        get_info<info::kernel_device_specific::work_group_size>(Dev);          \
    const auto DeviceMaxWorkItemSizes =                                        \
        Dev.get_info<info::device::max_work_item_sizes<Num>>();                \
    return generate_id<Num>(DeviceMaxWorkItemSizes, DeviceWgSize);             \
  } // namespace detail

ADD_TEMPLATE_METHOD_SPEC(1)
ADD_TEMPLATE_METHOD_SPEC(2)
ADD_TEMPLATE_METHOD_SPEC(3)

#undef ADD_TEMPLATE_METHOD_SPEC

#endif // __INTEL_PREVIEW_BREAKING_CHANGES

#define ADD_TEMPLATE_METHOD_SPEC(QueueSpec, Num, Kind, Reg)                    \
  template <>                                                                  \
  inline typename syclex::info::kernel_queue_specific::QueueSpec::return_type  \
  kernel_impl::ext_oneapi_get_info<                                            \
      syclex::info::kernel_queue_specific::QueueSpec>(                         \
      queue Queue, const range<Num> &WG) const {                               \
    if (WG.size() == 0)                                                        \
      throw exception(sycl::make_error_code(errc::invalid),                    \
                      "The work-group size cannot be zero.");                  \
    adapter_impl &Adapter = getAdapter();                                      \
    const auto DeviceNativeHandle =                                            \
        getSyclObjImpl(Queue.get_device())->getHandleRef();                    \
    uint32_t KernelSubWGSize = 0;                                              \
    Adapter.call<UrApiKind::Kind>(MKernel, DeviceNativeHandle, Reg,            \
                                  sizeof(uint32_t), &KernelSubWGSize,          \
                                  nullptr);                                    \
    return KernelSubWGSize;                                                    \
  }

ADD_TEMPLATE_METHOD_SPEC(max_sub_group_size, 3, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE)
ADD_TEMPLATE_METHOD_SPEC(max_sub_group_size, 2, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE)
ADD_TEMPLATE_METHOD_SPEC(max_sub_group_size, 1, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE)

ADD_TEMPLATE_METHOD_SPEC(num_sub_groups, 3, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS)
ADD_TEMPLATE_METHOD_SPEC(num_sub_groups, 2, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS)
ADD_TEMPLATE_METHOD_SPEC(num_sub_groups, 1, urKernelGetSubGroupInfo,
                         UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS)

#undef ADD_TEMPLATE_METHOD_SPEC
} // namespace detail
} // namespace _V1
} // namespace sycl
