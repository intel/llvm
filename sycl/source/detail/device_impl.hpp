//==----------------- device_impl.hpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/aspects.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/kernel_bundle.hpp>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace sycl {
inline namespace _V1 {

// Forward declaration
class platform;

namespace detail {

// Note that UR's enums have weird *_FORCE_UINT32 values, we ignore them in the
// callers. But we also can't write a fully-covered switch without mentioning it
// there, which wouldn't make any sense. As such, ensure that "real" values
// match and then just `static_cast` them (in the caller).
template <typename T0, typename T1>
constexpr bool enums_match(std::initializer_list<T0> l0,
                           std::initializer_list<T1> l1) {
  using U0 = std::underlying_type_t<T0>;
  using U1 = std::underlying_type_t<T1>;
  using C = std::common_type_t<U0, U1>;
  // std::equal isn't constexpr until C++20.
  if (l0.size() != l1.size())
    return false;
  auto i0 = l0.begin();
  auto e = l0.end();
  auto i1 = l1.begin();
  for (; i0 != e; ++i0, ++i1)
    if (static_cast<C>(*i0) != static_cast<C>(*i1))
      return false;
  return true;
}

// Forward declaration
class platform_impl;

// This could be a private member of the class, but old gcc can't handle that.
template <typename> static constexpr bool is_std_vector_v = false;
template <typename T>
static constexpr bool is_std_vector_v<std::vector<T>> = true;

template <ur_device_info_t Desc> static constexpr auto ur_ret_type_impl() {
  if constexpr (false) {
  }
#define MAP(VALUE, ...) else if constexpr (Desc == VALUE) return __VA_ARGS__{};
#include "ur_device_info_ret_types.inc"
#undef MAP
}

template <ur_device_info_t Desc>
using ur_ret_type = decltype(ur_ret_type_impl<Desc>());

// TODO: Make code thread-safe
class device_impl : public std::enable_shared_from_this<device_impl> {
  struct private_tag {
    explicit private_tag() = default;
  };
  friend class platform_impl;

  bool has_info_desc(ur_device_info_t Desc) const {
    size_t return_size = 0;
    return getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
               MDevice, Desc, 0, nullptr, &return_size) == UR_RESULT_SUCCESS;
  }

  // This should really be
  //   std::expected<ReturnT, ur_result_t>
  // but we don't have C++23. Emulate close enough with as little code as
  // possible.
  template <typename T, typename E>
  struct expected : public std::variant<T, E> {
    using base = std::variant<T, E>;
    using base::base;

    bool has_val() const { return this->index() == 0; }
    template <typename U> T value_or(U &&default_value) const {
      if (auto *p = std::get_if<0>(static_cast<const base *>(this)))
        return *p;
      else
        return std::forward<U>(default_value);
    }
    template <typename G> E error_or(G &&default_error) const {
      if (auto *p = std::get_if<1>(static_cast<const base *>(this)))
        return *p;
      else
        return std::forward<G>(default_error);
    }
    T value() const { return std::get<0>(*static_cast<const base *>(this)); }
    E error() const { return std::get<1>(*static_cast<const base *>(this)); }
  };

  template <ur_device_info_t Desc>
  expected<ur_ret_type<Desc>, ur_result_t> get_info_impl_nocheck() const {
    using ur_ret_t = ur_ret_type<Desc>;
    static_assert(!std::is_same_v<ur_ret_t, std::string>,
                  "Wasn't needed before.");
    if constexpr (is_std_vector_v<ur_ret_t>) {
      static_assert(
          !check_type_in_v<typename ur_ret_t::value_type, bool, std::string>);
      size_t ResultSize = 0;
      ur_result_t Error =
          getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
              getHandleRef(), Desc, 0, nullptr, &ResultSize);
      if (Error != UR_RESULT_SUCCESS)
        return {Error};
      if (ResultSize == 0)
        return {ur_ret_t{}};

      ur_ret_t Result(ResultSize / sizeof(typename ur_ret_t::value_type));
      Error = getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
          getHandleRef(), Desc, ResultSize, Result.data(), nullptr);
      if (Error != UR_RESULT_SUCCESS)
        return {Error};
      return {Result};
    } else {
      ur_ret_t Result;
      ur_result_t Error =
          getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
              getHandleRef(), Desc, sizeof(Result), &Result, nullptr);
      if (Error == UR_RESULT_SUCCESS)
        return {Result};
      else
        return {Error};
    }
  }

  template <ur_device_info_t Desc, bool InitializingCache = false>
  decltype(auto) get_info_impl() const {
    if constexpr (decltype(MCache)::has<URDesc<Desc>>() && !InitializingCache) {
      return MCache.get<URDesc<Desc>>();
    } else {
      using ur_ret_t = ur_ret_type<Desc>;
      if constexpr (std::is_same_v<ur_ret_t, std::string>) {
        return urGetInfoString<UrApiKind::urDeviceGetInfo>(*this, Desc);
      } else if constexpr (is_std_vector_v<ur_ret_t>) {
        size_t ResultSize = 0;
        getAdapter()->call<UrApiKind::urDeviceGetInfo>(getHandleRef(), Desc, 0,
                                                       nullptr, &ResultSize);
        if (ResultSize == 0)
          return ur_ret_t{};

        ur_ret_t Result(ResultSize / sizeof(typename ur_ret_t::value_type));
        getAdapter()->call<UrApiKind::urDeviceGetInfo>(
            getHandleRef(), Desc, ResultSize, Result.data(), nullptr);
        return Result;
      } else {
        ur_ret_t Result;
        getAdapter()->call<UrApiKind::urDeviceGetInfo>(
            getHandleRef(), Desc, sizeof(Result), &Result, nullptr);
        return Result;
      }
    }
  }

  // Define some helpers to cache properties. We use the same template
  // implementation for both SYCL information descriptors and raw calls to
  // `urDeviceGetInfo` by wrapping latter's `ur_device_info_t Desc` into a
  // wrapper class (to go from values to types, as we don't have universal
  // template parameters yet).
  //
  // Note that some modifications are also done in `get_info` and
  // `get_info_impl` so this caching is a part of device_impl implementation and
  // all the infrastructure should legitimally be as a class member.
  //
  // See `MCache` data member below for instruction how to make a property
  // cached.

  // Eager - initialize the value right in the device_impl's ctor.
  template <typename Desc> struct EagerCached {
    const typename Desc::return_type value;
  };

  // We optimize `init` signature so that it could be immediately
  // passed to `std::call_once` in the `CallOnceCached` below with an
  // expectation that it's easier to inline this lambda than outline a
  // creation of lambda in `CallOnceCached` if we'd be forced to have
  // one if `init` returned by value.
  //
  // Can't be an inline lambda because old gcc had a bug:
  // https://godbolt.org/z/h5K9TYceK.
  template <typename Desc, typename Initializer>
  static typename Desc::return_type getInitValue(device_impl &device) {
    typename Desc::return_type value;
    Initializer::template init<Desc>(device, value);
    return value;
  }

  template <typename Initializer, typename... Descs>
  struct EagerCache : EagerCached<Descs>... {
    EagerCache(device_impl &device)
        : EagerCached<Descs>{getInitValue<Descs, Initializer>(device)}... {}

    template <typename Desc> static constexpr bool has() {
      return ((std::is_same_v<Desc, Descs> || ...));
    }

    template <typename Desc> decltype(auto) get() const {
      // Extra parentheses to return as reference (see `decltype(auto)`).
      return (static_cast<const EagerCached<Desc> *>(this)->value);
    }
  };

#if defined(_GLIBCXX_RELEASE)
  // libstdc++'s std::call_once is significantly slower than libc++
  // implementation (30-40ns for libc++ CallOnceCache/EagerCache vs 50-60ns for
  // CallOnceCache when using libstdc++ for queries of simple types like
  // `ur_device_usm_access_capability_flags_t`). libc++ implements it via
  // `__cxa_guard_*` (same as function static variables initialization) but
  // libstdc++ cannot do that without an ABI break:
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66146#c53.
  //
  // We do care about performance of the fast path and can pay extra costs in
  // memory/slow-down during single init call, so add an extra flag to optimize.
#define GUARD_STD_CALL_ONCE_WITH_EXTRA_CHECK 1
#else
#define GUARD_STD_CALL_ONCE_WITH_EXTRA_CHECK 0
#endif

  // CallOnce - initialize on first query, but exactly once so that we could
  // return cached values by reference. Important for `std::vector` /
  // `std::string` values where returning cached values by value would cause
  // heap allocations.
  template <typename Desc> struct CallOnceCached {
    std::once_flag flag;
    typename Desc::return_type value;
#if GUARD_STD_CALL_ONCE_WITH_EXTRA_CHECK
    std::atomic_bool initialized = false;
#endif
  };

  template <typename Initializer, typename... Descs>
  struct CallOnceCache : public CallOnceCached<Descs>... {
    device_impl &device;

    CallOnceCache(device_impl &device) : device(device) {}

    template <typename Desc> static constexpr bool has() {
      return ((std::is_same_v<Desc, Descs> || ...));
    }

    template <typename Desc> decltype(auto) get() {
      auto &Entry = *static_cast<CallOnceCached<Desc> *>(this);
#if GUARD_STD_CALL_ONCE_WITH_EXTRA_CHECK
      if (!Entry.initialized.load(std::memory_order_acquire)) {
        std::call_once(Entry.flag, [&]() {
          Initializer::template init<Desc>(device, Entry.value);
          Entry.initialized.store(true, std::memory_order_release);
        });
      }
#else
      std::call_once(Entry.flag, Initializer::template init<Desc>, device,
                     Entry.value);
#endif
      // Extra parentheses to return as reference (see `decltype(auto)`).
      return (std::as_const(Entry.value));
    }
  };

#undef GUARD_STD_CALL_ONCE_WITH_EXTRA_CHECK

  // get_info and get_info_impl need to know if a particular query is cacheable.
  // It's easier if all the cache instances (eager/call-once * UR/SYCL) are
  // merged into a single object.
  template <typename... Caches> struct JointCache : public Caches... {
    JointCache(device_impl &device) : Caches(device)... {}

    template <typename Desc> static constexpr bool has() {
      // GCC 7.* had a bug: https://godbolt.org/z/cKeoTqMba, workaround it by
      // not performing extra checks. Builds with unaffected compilers would
      // catch all the issues.
#if !(defined(__GNUC__) && !defined(__clang__)) || (__GNUC__ >= 8)
      constexpr int NumFound = []() constexpr {
        int found = 0;
        (((found = Caches::template has<Desc>() ? found + 1 : found), ...));
        return found;
      }();
      static_assert(NumFound <= 1,
                    "Multiple caches must not contain the same descriptor");
      return NumFound == 1;
#else
      return ((Caches::template has<Desc>() || ...));
#endif
    }

    template <typename Desc> decltype(auto) get() {
      static_assert(has<Desc>());
      constexpr auto Idx = []() constexpr {
        int i = 0;
        int found = 0;
        (((found = Caches::template has<Desc>() ? i : found, ++i), ...));
        return found;
      }();
      return nth_type_t<Idx, Caches...>::template get<Desc>();
    }

    //  Can't provide `has<ur_device_info_t>`/`has<aspect>` (or similar for
    //  `get`) due to MSVC bug: https://godbolt.org/z/s6bP6qK4f.
  };

  // With generic infrastructure above finished, provide the customization
  // points:

  struct InfoInitializer {
    template <typename Desc>
    static void init(device_impl &device, typename Desc::return_type &value) {
      value = device.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
              get_info
#else
              get_info_abi_workaround
#endif
              <Desc, true /* InitializingCache */>();
    }
  };

  template <ur_device_info_t Desc> struct URDesc {
    using return_type = ur_ret_type<Desc>;
    static constexpr ur_device_info_t UR_DESC = Desc;
  };

  struct URInfoInitializer {
    template <typename Desc>
    static void init(device_impl &device, typename Desc::return_type &value) {
      value =
          device.get_info_impl<Desc::UR_DESC, true /* InitializingCache */>();
    }
  };

  template <template <typename...> typename Cache, ur_device_info_t... Descs>
  using URCache = Cache<URInfoInitializer, URDesc<Descs>...>;

  template <ur_device_info_t... Descs>
  using UREagerCache = URCache<EagerCache, Descs...>;
  template <ur_device_info_t... Descs>
  using URCallOnceCache = URCache<CallOnceCache, Descs...>;

  template <aspect Aspect_> struct AspectDesc {
    using return_type = bool;
    static constexpr aspect Aspect = Aspect_;
  };

  struct AspectInitializer {
    template <typename AspectDesc>
    static void init(device_impl &device, bool &value) {
      value = device.has<AspectDesc::Aspect, true /* InitializingCache */>();
    }
  };

  template <template <typename...> typename Cache, aspect... Aspects>
  using AspectCache = Cache<AspectInitializer, AspectDesc<Aspects>...>;

public:
  /// Constructs a SYCL device instance using the provided
  /// UR device instance.
  //
  // Must be called through `platform_impl::getOrMakeDeviceImpl` only.
  // `private_tag` ensures that is true.
  explicit device_impl(ur_device_handle_t Device, platform_impl &Platform,
                       private_tag);

  ~device_impl();

  /// Get instance of OpenCL device
  ///
  /// \return a valid cl_device_id instance in accordance with the
  /// requirements described in 4.3.1.
  cl_device_id get() const;

  /// Get reference to UR device
  ///
  /// For host device an exception is thrown
  ///
  /// \return non-constant reference to UR device
  ur_device_handle_t &getHandleRef() { return MDevice; }

  /// Get constant reference to UR device
  ///
  /// For host device an exception is thrown
  ///
  /// \return constant reference to UR device
  const ur_device_handle_t &getHandleRef() const { return MDevice; }

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool is_cpu() const {
    return get_info_impl<UR_DEVICE_INFO_TYPE>() == UR_DEVICE_TYPE_CPU;
  }

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool is_gpu() const {
    return get_info_impl<UR_DEVICE_INFO_TYPE>() == UR_DEVICE_TYPE_GPU;
  }

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool is_accelerator() const {
    return get_info_impl<UR_DEVICE_INFO_TYPE>() == UR_DEVICE_TYPE_FPGA;
  }

  /// Get associated SYCL platform
  ///
  /// If this SYCL device is an OpenCL device then the SYCL platform
  /// must encapsulate the OpenCL cl_plaform_id associated with the
  /// underlying OpenCL cl_device_id of this SYCL device. If this SYCL device
  /// is a host device then the SYCL platform must be a host platform.
  /// The value returned must be equal to that returned
  /// by get_info<info::device::platform>().
  ///
  /// \return The associated SYCL platform.
  platform get_platform() const;

  /// \return the associated adapter with this device.
  const AdapterPtr &getAdapter() const { return MPlatform->getAdapter(); }

  /// Check SYCL extension support by device
  ///
  /// \param ExtensionName is a name of queried extension.
  /// \return true if SYCL device supports the extension.
  bool has_extension(const std::string &ExtensionName) const;

  std::vector<device>
  create_sub_devices(const ur_device_partition_properties_t *Properties,
                     size_t SubDevicesCount) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_equally a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param ComputeUnits is a desired count of compute units in each sub
  /// device.
  /// \return A vector class of sub devices partitioned equally from this
  /// SYCL device based on the ComputeUnits parameter.
  std::vector<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param Counts is a std::vector of desired compute units in sub devices.
  /// \return a std::vector of sub devices partitioned from this SYCL device
  /// by count sizes based on the Counts parameter.
  std::vector<device>
  create_sub_devices(const std::vector<size_t> &Counts) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_affinity_domain or the SYCL
  /// device does not support info::affinity_domain provided a
  /// feature_not_supported exception must be thrown.
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of
  /// SYCL Spec
  /// \return a vector class of sub devices partitioned from this SYCL device
  /// by affinity domain based on the AffinityDomain parameter
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::ext_intel_partition_by_cslice a
  /// feature_not_supported exception must be thrown.
  ///
  /// \return a vector class of sub devices partitioned from this SYCL
  /// device at a granularity of "cslice" (compute slice).
  std::vector<device> create_sub_devices() const;

  /// Check if desired partition property supported by device
  ///
  /// \param Prop is one of info::partition_property::(partition_equally,
  /// partition_by_counts, partition_by_affinity_domain)
  /// \return true if Prop is supported by device.
  bool is_partition_supported(info::partition_property Prop) const;

  /// Queries this SYCL device for information requested by the template
  /// parameter param
  ///
  /// Specializations of info::param_traits must be defined in accordance
  /// with the info parameters in Table 4.20 of SYCL Spec to facilitate
  /// returning the type associated with the param parameter.
  ///
  /// \return device info of type described in Table 4.20.

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  template <typename Param, bool InitializingCache = false>
  decltype(auto) get_info() const {
#define CALL_GET_INFO get_info
#else
  // We've been exporting
  // `device_impl::get_info<ext::<whatever>::info::device::<descriptor>` for no
  // reason. Have to keep doing that until next ABI breaking window. Also, old
  // gcc doesn't allow in-class specializations, so they have to go out-of-class
  // which happens later then implicit instantiatons of delegating to
  // `get_info<other_descriptor>`. As such, all such calls have to go through
  // `get_info_abi_workaround` for which we need this ugly macro:
#define CALL_GET_INFO get_info_abi_workaround
  template <typename Param> typename Param::return_type get_info() const;
  template <typename Param, bool InitializingCache = false>
  decltype(auto) get_info_abi_workaround() const {
#endif
    using execution_scope = ext::oneapi::experimental::execution_scope;

    // With the return type of this function being automatically
    // deduced we can't simply do
    //
    //    CASE(Desc1) { return get_info<Desc2>(); }
    //
    // because the function isn't defined yet and we can't auto-deduce the
    // return type for `Desc2` yet. The solution here is to make that delegation
    // template-parameter-dependent. We use the `InitializingCache` parameter
    // for that out of convenience.
    //
    // Note that for "eager" cache it's the programmer's responsibility that
    // the descriptor we delegate to is initialized first (by referencing that
    // descriptor first when defining the cache data member). For "CallOnce"
    // cache we want to be querying cached value so "false" is the right
    // template parameter for such delegation.
    [[maybe_unused]] constexpr bool DependentFalse = InitializingCache && false;

    if constexpr (decltype(MCache)::has<Param>() && !InitializingCache) {
      return MCache.get<Param>();
    }
#define CASE(PARAM) else if constexpr (std::is_same_v<Param, PARAM>)
    // device_traits.def

    CASE(info::device::device_type) {
      using device_type = info::device_type;
      switch (get_info_impl<UR_DEVICE_INFO_TYPE>()) {
      case UR_DEVICE_TYPE_DEFAULT:
        return device_type::automatic;
      case UR_DEVICE_TYPE_ALL:
        return device_type::all;
      case UR_DEVICE_TYPE_GPU:
        return device_type::gpu;
      case UR_DEVICE_TYPE_CPU:
        return device_type::cpu;
      case UR_DEVICE_TYPE_FPGA:
        return device_type::accelerator;
      case UR_DEVICE_TYPE_MCA:
      case UR_DEVICE_TYPE_VPU:
        return device_type::custom;
      default: {
        assert(false);
        // FIXME: what is that???
        return device_type::custom;
      }
      }
    }

    CASE(info::device::max_work_item_sizes<3>) {
      auto result = get_info_impl<UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES>();
      return range<3>{result[2], result[1], result[0]};
    }
    CASE(info::device::max_work_item_sizes<2>) {
      range<3> r3 =
          CALL_GET_INFO<info::device::max_work_item_sizes<3>, DependentFalse>();
      return range<2>{r3[1], r3[2]};
    }
    CASE(info::device::max_work_item_sizes<1>) {
      range<3> r3 =
          CALL_GET_INFO<info::device::max_work_item_sizes<3>, DependentFalse>();
      return range<1>{r3[2]};
    }

    CASE(info::device::sub_group_sizes) {
      std::vector<uint32_t> ur_result =
          get_info_impl<UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL>();
      std::vector<size_t> result;
      result.reserve(ur_result.size());
      std::copy(ur_result.begin(), ur_result.end(), std::back_inserter(result));
      return result;
    }

    CASE(info::device::single_fp_config) {
      return get_fp_config<UR_DEVICE_INFO_SINGLE_FP_CONFIG>();
    }
    CASE(info::device::half_fp_config) {
      return get_fp_config<UR_DEVICE_INFO_HALF_FP_CONFIG>();
    }
    CASE(info::device::double_fp_config) {
      return get_fp_config<UR_DEVICE_INFO_DOUBLE_FP_CONFIG>();
    }

    CASE(info::device::global_mem_cache_type) {
      using cache = info::global_mem_cache_type;
      static_assert(
          enums_match({UR_DEVICE_MEM_CACHE_TYPE_NONE,
                       UR_DEVICE_MEM_CACHE_TYPE_READ_ONLY_CACHE,
                       UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE},
                      {cache::none, cache::read_only, cache::read_write}));
      return static_cast<cache>(
          get_info_impl<UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE>());
    }

    CASE(info::device::local_mem_type) {
      using mem = info::local_mem_type;
      static_assert(enums_match({UR_DEVICE_LOCAL_MEM_TYPE_NONE,
                                 UR_DEVICE_LOCAL_MEM_TYPE_LOCAL,
                                 UR_DEVICE_LOCAL_MEM_TYPE_GLOBAL},
                                {mem::none, mem::local, mem::global}));
      return static_cast<mem>(get_info_impl<UR_DEVICE_INFO_LOCAL_MEM_TYPE>());
    }

    CASE(info::device::atomic_memory_order_capabilities) {
      return readMemoryOrderBitfield(
          get_info_impl<UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES>());
    }
    CASE(info::device::atomic_fence_order_capabilities) {
      return readMemoryOrderBitfield(
          get_info_impl<UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES>());
    }
    CASE(info::device::atomic_memory_scope_capabilities) {
      return readMemoryScopeBitfield(
          get_info_impl<UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES>());
    }
    CASE(info::device::atomic_fence_scope_capabilities) {
      return readMemoryScopeBitfield(
          get_info_impl<UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES>());
    }

    CASE(info::device::execution_capabilities) {
      if (getBackend() != backend::opencl)
        throw exception(make_error_code(errc::invalid),
                        "info::device::execution_capabilities is available for "
                        "backend::opencl only");

      ur_device_exec_capability_flags_t bits =
          get_info_impl<UR_DEVICE_INFO_EXECUTION_CAPABILITIES>();
      std::vector<info::execution_capability> result;
      if (bits & UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL)
        result.push_back(info::execution_capability::exec_kernel);
      if (bits & UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL)
        result.push_back(info::execution_capability::exec_native_kernel);
      return result;
    }

    CASE(info::device::queue_profiling) {
      // Specialization for queue_profiling. In addition to ur_queue level
      // profiling, urDeviceGetGlobalTimestamps is not supported,
      // command_submit, command_start, command_end will be calculated. See
      // MFallbackProfiling
      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_QUEUE_PROPERTIES>() &
          UR_QUEUE_FLAG_PROFILING_ENABLE);
    }

    CASE(info::device::built_in_kernels) {
      return split_string(get_info_impl<UR_DEVICE_INFO_BUILT_IN_KERNELS>(),
                          ';');
    }
    CASE(info::device::built_in_kernel_ids) {
      auto names =
          CALL_GET_INFO<info::device::built_in_kernels, DependentFalse>();

      std::vector<kernel_id> ids;
      ids.reserve(names.size());

      auto &PM = ProgramManager::getInstance();
      for (const auto &name : names)
        ids.push_back(PM.getBuiltInKernelID(name));

      return ids;
    }

    CASE(info::device::platform) {
      return createSyclObjFromImpl<platform>(
          platform_impl::getOrMakePlatformImpl(
              get_info_impl<UR_DEVICE_INFO_PLATFORM>(), getAdapter()));
    }

    CASE(info::device::profile) {
      if (getBackend() != backend::opencl)
        throw sycl::exception(errc::invalid,
                              "the info::device::profile info descriptor can "
                              "only be queried with an OpenCL backend");

      return get_info_impl<UR_DEVICE_INFO_PROFILE>();
    }

    CASE(info::device::extensions) {
      return split_string(get_info_impl<UR_DEVICE_INFO_EXTENSIONS>(), ' ');
    }

    CASE(info::device::preferred_interop_user_sync) {
      if (getBackend() != backend::opencl)
        throw sycl::exception(
            errc::invalid,
            "the info::device::preferred_interop_user_sync info descriptor can "
            "only be queried with an OpenCL backend");

      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC>());
    }

    CASE(info::device::partition_properties) {
      std::vector<ur_device_partition_t> ur_dev_partitions =
          get_info_impl<UR_DEVICE_INFO_SUPPORTED_PARTITIONS>();
      std::vector<info::partition_property> result;
      result.reserve(ur_dev_partitions.size());
      for (auto &entry : ur_dev_partitions) {
        // OpenCL extensions may have partition_properties that
        // are not yet defined for SYCL (eg. CL_DEVICE_PARTITION_BY_NAMES_INTEL)
        info::partition_property pp(info::ConvertPartitionProperty(entry));
        switch (pp) {
        case info::partition_property::no_partition:
        case info::partition_property::partition_equally:
        case info::partition_property::partition_by_counts:
        case info::partition_property::partition_by_affinity_domain:
        case info::partition_property::ext_intel_partition_by_cslice:
          result.push_back(pp);
        }
      }

      return result;
    }
    CASE(info::device::partition_affinity_domains) {
      ur_device_affinity_domain_flags_t bits =
          get_info_impl<UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN>();
      std::vector<info::partition_affinity_domain> result;
      using domain = info::partition_affinity_domain;
      constexpr std::pair<ur_device_affinity_domain_flags_t, domain> mapping[] =
          {{UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA, domain::numa},
           {UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE, domain::L4_cache},
           {UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE, domain::L3_cache},
           {UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE, domain::L2_cache},
           {UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE, domain::L1_cache},
           {UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
            domain::next_partitionable}};
      for (auto [k, v] : mapping)
        if (bits & k)
          result.push_back(v);

      return result;
    }
    CASE(info::device::partition_type_property) {
      std::vector<ur_device_partition_property_t> PartitionProperties =
          get_info_impl<UR_DEVICE_INFO_PARTITION_TYPE>();
      if (PartitionProperties.empty())
        return info::partition_property::no_partition;
      // The old UR implementation also just checked the first element, is that
      // correct?
      return info::ConvertPartitionProperty(PartitionProperties[0].type);
    }
    CASE(info::device::partition_type_affinity_domain) {
      std::vector<ur_device_partition_property_t> PartitionProperties =
          get_info_impl<UR_DEVICE_INFO_PARTITION_TYPE>();
      if (PartitionProperties.empty())
        return info::partition_affinity_domain::not_applicable;
      for (const auto &PartitionProp : PartitionProperties) {
        if (PartitionProp.type == UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)
          return info::ConvertAffinityDomain(
              PartitionProp.value.affinity_domain);
      }

      return info::partition_affinity_domain::not_applicable;
    }

    CASE(info::device::parent_device) {
      auto ur_parent_dev = get_info_impl<UR_DEVICE_INFO_PARENT_DEVICE>();
      if (ur_parent_dev == nullptr)
        throw exception(make_error_code(errc::invalid),
                        "No parent for device because it is not a subdevice");

      return createSyclObjFromImpl<device>(
          getPlatformImpl().getOrMakeDeviceImpl(ur_parent_dev));
    }

    CASE(info::device::image_support) {
      // No devices currently support SYCL 2020 images.
      return false;
    }

    CASE(info::device::kernel_kernel_pipe_support) {
      // We claim, that all Intel FPGA devices support kernel to kernel pipe
      // feature (at least at the scope of SYCL_INTEL_data_flow_pipes
      // extension).
      std::string platform_name = MPlatform->get_info<info::platform::name>();
      if (platform_name == "Intel(R) FPGA Emulation Platform for OpenCL(TM)" ||
          platform_name == "Intel(R) FPGA SDK for OpenCL(TM)")
        return true;

      // TODO: a better way is to query for supported SPIR-V capabilities when
      // it's started to be possible. Also, if a device's backend supports
      // SPIR-V 1.1 (where Pipe Storage feature was defined), than it supports
      // the feature as well.
      return false;
    }

    CASE(info::device::usm_device_allocations) {
      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_USM_DEVICE_SUPPORT>() &
          UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
    }
    CASE(info::device::usm_host_allocations) {
      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_USM_HOST_SUPPORT>() &
          UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
    }
    CASE(info::device::usm_shared_allocations) {
      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT>() &
          UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
    }
    CASE(info::device::usm_restricted_shared_allocations) {
      ur_device_usm_access_capability_flags_t cap_flags =
          get_info_impl<UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT>();
      // Check that we don't support any cross device sharing
      return !(cap_flags &
               (UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
                UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS));
    }
    CASE(info::device::usm_system_allocations) {
      return static_cast<bool>(
          get_info_impl<UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT>() &
          UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
    }

    CASE(info::device::opencl_c_version) {
      throw sycl::exception(errc::feature_not_supported,
                            "Deprecated interface that hasn't been working for "
                            "some time already");
      return std::string{}; // for return type deduction.
    }

    CASE(ext::intel::info::device::max_mem_bandwidth) {
      if (!has(aspect::ext_intel_max_mem_bandwidth))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_max_mem_bandwidth aspect");
      return get_info_impl<UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH>();
    }

    CASE(info::device::ext_oneapi_max_global_work_groups) {
      // Deprecated alias.
      return CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_global_work_groups,
          DependentFalse>();
    }
    CASE(info::device::ext_oneapi_max_work_groups_1d) {
      // Deprecated alias.
      return CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_work_groups<1>,
          DependentFalse>();
    }
    CASE(info::device::ext_oneapi_max_work_groups_2d) {
      // Deprecated alias.
      return CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_work_groups<2>,
          DependentFalse>();
    }
    CASE(info::device::ext_oneapi_max_work_groups_3d) {
      // Deprecated alias.
      return CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_work_groups<3>,
          DependentFalse>();
    }

    CASE(info::device::ext_oneapi_cuda_cluster_group) {
      auto SupportFlags =
          get_info_impl<UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES>();
      return static_cast<bool>(
          SupportFlags & UR_KERNEL_LAUNCH_PROPERTIES_FLAG_CLUSTER_DIMENSION);
    }

    // ext_codeplay_device_traits.def

    CASE(ext::codeplay::experimental::info::device::supports_fusion) {
      // TODO(#15184): Remove this aspect in the next ABI-breaking window.
      return false;
    }

    // ext_oneapi_device_traits.def

    CASE(ext::oneapi::experimental::info::device::max_global_work_groups) {
      return static_cast<size_t>((std::numeric_limits<int>::max)());
    }
    CASE(ext::oneapi::experimental::info::device::max_work_groups<3>) {
      size_t Limit = CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_global_work_groups,
          DependentFalse>();

      // TODO: std::array<size_t, 3> ?
      size_t result[3];
      getAdapter()->call<UrApiKind::urDeviceGetInfo>(
          getHandleRef(), UR_DEVICE_INFO_MAX_WORK_GROUPS_3D, sizeof(result),
          &result, nullptr);
      return id<3>(std::min(Limit, result[2]), std::min(Limit, result[1]),
                   std::min(Limit, result[0]));
    }
    CASE(ext::oneapi::experimental::info::device::max_work_groups<2>) {
      id<3> max_3d = CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_work_groups<3>,
          DependentFalse>();
      return id<2>{max_3d[1], max_3d[2]};
    }
    CASE(ext::oneapi::experimental::info::device::max_work_groups<1>) {
      id<3> max_3d = CALL_GET_INFO<
          ext::oneapi::experimental::info::device::max_work_groups<3>,
          DependentFalse>();
      return id<1>{max_3d[2]};
    }

    CASE(ext::oneapi::experimental::info::device::
             work_group_progress_capabilities<execution_scope::root_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::work_group, execution_scope::root_group));
    }
    CASE(ext::oneapi::experimental::info::device::
             sub_group_progress_capabilities<execution_scope::root_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::sub_group, execution_scope::root_group));
    }
    CASE(ext::oneapi::experimental::info::device::
             sub_group_progress_capabilities<execution_scope::work_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::sub_group, execution_scope::work_group));
    }
    CASE(ext::oneapi::experimental::info::device::
             work_item_progress_capabilities<execution_scope::root_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::work_item, execution_scope::root_group));
    }
    CASE(ext::oneapi::experimental::info::device::
             work_item_progress_capabilities<execution_scope::work_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::work_item, execution_scope::work_group));
    }
    CASE(ext::oneapi::experimental::info::device::
             work_item_progress_capabilities<
                 ext::oneapi::experimental::execution_scope::sub_group>) {
      return getProgressGuaranteesUpTo(getProgressGuarantee(
          execution_scope::work_item, execution_scope::sub_group));
    }

    CASE(ext::oneapi::experimental::info::device::architecture) {
      return get_architecture();
    }

    CASE(ext::oneapi::experimental::info::device::matrix_combinations) {
      return get_matrix_combinations();
    }

    CASE(ext::oneapi::experimental::info::device::mipmap_max_anisotropy) {
      return static_cast<float>(
          get_info_impl<UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP>());
    }

    CASE(ext::oneapi::experimental::info::device::component_devices) {
      expected<std::vector<ur_device_handle_t>, ur_result_t> Devs =
          get_info_impl_nocheck<UR_DEVICE_INFO_COMPONENT_DEVICES>();
      if (!Devs.has_val()) {
        ur_result_t Err = Devs.error();
        if (Err == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION)
          return std::vector<sycl::device>{};
        getAdapter()->checkUrResult(Err);
      }

      std::vector<sycl::device> Result;
      Result.reserve(Devs.value().size());
      for (const auto &d : Devs.value())
        Result.push_back(
            createSyclObjFromImpl<device>(MPlatform->getOrMakeDeviceImpl(d)));

      return Result;
    }
    CASE(ext::oneapi::experimental::info::device::composite_device) {
      if (!has(sycl::aspect::ext_oneapi_is_component))
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Only devices with aspect::ext_oneapi_is_component "
            "can call this function.");

      if (ur_device_handle_t Result =
              get_info_impl<UR_DEVICE_INFO_COMPOSITE_DEVICE>())
        return createSyclObjFromImpl<device>(
            MPlatform->getOrMakeDeviceImpl(Result));

      throw sycl::exception(make_error_code(errc::invalid),
                            "A component with aspect::ext_oneapi_is_component "
                            "must have a composite device.");
    }
    CASE(ext::oneapi::info::device::num_compute_units) {
      return static_cast<size_t>(
          get_info_impl<UR_DEVICE_INFO_NUM_COMPUTE_UNITS>());
    }

    // ext_intel_device_traits.def

    CASE(ext::intel::info::device::device_id) {
      if (!has(aspect::ext_intel_device_id))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_device_id aspect");
      return get_info_impl<UR_DEVICE_INFO_DEVICE_ID>();
    }
    CASE(ext::intel::info::device::pci_address) {
      if (!has(aspect::ext_intel_pci_address))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_pci_address aspect");
      return get_info_impl<UR_DEVICE_INFO_PCI_ADDRESS>();
    }
    CASE(ext::intel::info::device::gpu_eu_count) {
      if (!has(aspect::ext_intel_gpu_eu_count))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_gpu_eu_count aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_EU_COUNT>();
    }
    CASE(ext::intel::info::device::gpu_eu_simd_width) {
      if (!has(aspect::ext_intel_gpu_eu_simd_width))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_gpu_eu_simd_width aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH>();
    }
    CASE(ext::intel::info::device::gpu_slices) {
      if (!has(aspect::ext_intel_gpu_slices))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_gpu_slices aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_EU_SLICES>();
    }
    CASE(ext::intel::info::device::gpu_subslices_per_slice) {
      if (!has(aspect::ext_intel_gpu_subslices_per_slice))
        throw exception(make_error_code(errc::feature_not_supported),
                        "The device does not have the "
                        "ext_intel_gpu_subslices_per_slice aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE>();
    }
    CASE(ext::intel::info::device::gpu_eu_count_per_subslice) {
      if (!has(aspect::ext_intel_gpu_eu_count_per_subslice))
        throw exception(make_error_code(errc::feature_not_supported),
                        "The device does not have the "
                        "ext_intel_gpu_eu_count_per_subslice aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE>();
    }
    CASE(ext::intel::info::device::gpu_hw_threads_per_eu) {
      if (!has(aspect::ext_intel_gpu_hw_threads_per_eu))
        throw exception(make_error_code(errc::feature_not_supported),
                        "The device does not have the "
                        "ext_intel_gpu_hw_threads_per_eu aspect");
      return get_info_impl<UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU>();
    }
    CASE(ext::intel::info::device::uuid) {
      if (!has(aspect::ext_intel_device_info_uuid))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_device_info_uuid aspect");
      // TODO: we're essentially memcpy'ing here...
      static_assert(std::is_same_v<uuid_type, std::array<unsigned char, 16>>);
      return get_info_impl<UR_DEVICE_INFO_UUID>();
    }
    CASE(ext::intel::info::device::free_memory) {
      if (!has(aspect::ext_intel_free_memory))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_free_memory aspect");
      return get_info_impl<UR_DEVICE_INFO_GLOBAL_MEM_FREE>();
    }
    CASE(ext::intel::info::device::memory_clock_rate) {
      if (!has(aspect::ext_intel_memory_clock_rate))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_memory_clock_rate aspect");
      return get_info_impl<UR_DEVICE_INFO_MEMORY_CLOCK_RATE>();
    }
    CASE(ext::intel::info::device::memory_bus_width) {
      if (!has(aspect::ext_intel_memory_bus_width))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_memory_bus_width aspect");
      return get_info_impl<UR_DEVICE_INFO_MEMORY_BUS_WIDTH>();
    }
    CASE(ext::intel::info::device::max_compute_queue_indices) {
      return static_cast<int>(
          get_info_impl<UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES>());
    }
    CASE(ext::intel::esimd::info::device::has_2d_block_io_support) {
      if (!has(aspect::ext_intel_esimd))
        return false;
      ur_exp_device_2d_block_array_capability_flags_t BlockArrayCapabilities =
          get_info_impl<UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP>();
      return (BlockArrayCapabilities &
              UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_LOAD) &&
             (BlockArrayCapabilities &
              UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_STORE);
    }
    CASE(ext::intel::info::device::current_clock_throttle_reasons) {
      if (!has(aspect::ext_intel_current_clock_throttle_reasons))
        throw exception(make_error_code(errc::feature_not_supported),
                        "The device does not have the "
                        "ext_intel_current_clock_throttle_reasons aspect");

      ur_device_throttle_reasons_flags_t UrThrottleReasons =
          get_info_impl<UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS>();
      std::vector<ext::intel::throttle_reason> ThrottleReasons;
      using reason = ext::intel::throttle_reason;
      constexpr std::pair<ur_device_throttle_reasons_flags_t, reason>
          UR2SYCLMappings[] = {
              {UR_DEVICE_THROTTLE_REASONS_FLAG_POWER_CAP, reason::power_cap},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_CURRENT_LIMIT,
               reason::current_limit},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_THERMAL_LIMIT,
               reason::thermal_limit},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_PSU_ALERT, reason::psu_alert},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_SW_RANGE, reason::sw_range},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_HW_RANGE, reason::hw_range},
              {UR_DEVICE_THROTTLE_REASONS_FLAG_OTHER, reason::other}};

      for (const auto &[UrFlag, SyclReason] : UR2SYCLMappings) {
        if (UrThrottleReasons & UrFlag) {
          ThrottleReasons.push_back(SyclReason);
        }
      }
      return ThrottleReasons;
    }
    CASE(ext::intel::info::device::fan_speed) {
      if (!has(aspect::ext_intel_fan_speed))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_fan_speed aspect");
      return get_info_impl<UR_DEVICE_INFO_FAN_SPEED>();
    }
    CASE(ext::intel::info::device::max_power_limit) {
      if (!has(aspect::ext_intel_power_limits))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_power_limits aspect");
      return get_info_impl<UR_DEVICE_INFO_MAX_POWER_LIMIT>();
    }
    CASE(ext::intel::info::device::min_power_limit) {
      if (!has(aspect::ext_intel_power_limits))
        throw exception(
            make_error_code(errc::feature_not_supported),
            "The device does not have the ext_intel_power_limits aspect");
      return get_info_impl<UR_DEVICE_INFO_MIN_POWER_LIMIT>();
    }
    else {
      constexpr auto Desc = UrInfoCode<Param>::value;
      return static_cast<typename Param::return_type>(get_info_impl<Desc>());
    }
#undef CASE
  }

  // template version is necessary to make this cacheable (cache lookup needs
  // compile-time data).
  template <aspect Aspect, bool InitializingCache = false> bool has() const {
    if constexpr (decltype(MCache)::has<AspectDesc<Aspect>>() &&
                  !InitializingCache) {
      return MCache.get<AspectDesc<Aspect>>();
    }
#define CASE(ASPECT) else if constexpr (Aspect == aspect::ASPECT)
    CASE(host) {
      // Deprecated
      return false;
    }
    CASE(cpu) { return is_cpu(); }
    CASE(gpu) { return is_gpu(); }
    CASE(accelerator) { return is_accelerator(); }
    CASE(custom) {
      return false;
      // TODO: Implement this for FPGA emulator.
    }
    CASE(emulated) { return false; }
    CASE(host_debuggable) { return false; }
    CASE(fp16) { return has_extension("cl_khr_fp16"); }
    CASE(fp64) { return has_extension("cl_khr_fp64"); }
    CASE(int64_base_atomics) {
      return has_extension("cl_khr_int64_base_atomics");
    }
    CASE(int64_extended_atomics) {
      return has_extension("cl_khr_int64_extended_atomics");
    }
    CASE(atomic64) { return get_info<info::device::atomic64>(); }
    CASE(image) { return get_info<info::device::image_support>(); }
    CASE(online_compiler) {
      return get_info<info::device::is_compiler_available>();
    }
    CASE(online_linker) {
      return get_info<info::device::is_linker_available>();
    }
    CASE(queue_profiling) { return get_info<info::device::queue_profiling>(); }
    CASE(usm_device_allocations) {
      return get_info<info::device::usm_device_allocations>();
    }
    CASE(usm_host_allocations) {
      return get_info<info::device::usm_host_allocations>();
    }
    CASE(ext_intel_mem_channel) {
      return get_info<info::device::ext_intel_mem_channel>();
    }
    CASE(ext_oneapi_cuda_cluster_group) {
      return get_info<info::device::ext_oneapi_cuda_cluster_group>();
    }
    CASE(usm_atomic_host_allocations) {
      return (get_info_impl<UR_DEVICE_INFO_USM_HOST_SUPPORT>() &
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS);
    }
    CASE(usm_shared_allocations) {
      return get_info<info::device::usm_shared_allocations>();
    }
    CASE(usm_atomic_shared_allocations) {
      return (get_info_impl<UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT>() &
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS);
    }
    CASE(usm_restricted_shared_allocations) {
      return get_info<info::device::usm_restricted_shared_allocations>();
    }
    CASE(usm_system_allocations) {
      return get_info<info::device::usm_system_allocations>();
    }
    CASE(ext_intel_device_id) {
      return has_info_desc(UR_DEVICE_INFO_DEVICE_ID);
    }
    CASE(ext_intel_pci_address) {
      return has_info_desc(UR_DEVICE_INFO_PCI_ADDRESS);
    }
    CASE(ext_intel_gpu_eu_count) {
      return has_info_desc(UR_DEVICE_INFO_GPU_EU_COUNT);
    }
    CASE(ext_intel_gpu_eu_simd_width) {
      return has_info_desc(UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH);
    }
    CASE(ext_intel_gpu_slices) {
      return has_info_desc(UR_DEVICE_INFO_GPU_EU_SLICES);
    }
    CASE(ext_intel_gpu_subslices_per_slice) {
      return has_info_desc(UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE);
    }
    CASE(ext_intel_gpu_eu_count_per_subslice) {
      return has_info_desc(UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE);
    }
    CASE(ext_intel_gpu_hw_threads_per_eu) {
      return has_info_desc(UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU);
    }
    CASE(ext_intel_free_memory) {
      return has_info_desc(UR_DEVICE_INFO_GLOBAL_MEM_FREE);
    }
    CASE(ext_intel_memory_clock_rate) {
      return has_info_desc(UR_DEVICE_INFO_MEMORY_CLOCK_RATE);
    }
    CASE(ext_intel_memory_bus_width) {
      return has_info_desc(UR_DEVICE_INFO_MEMORY_BUS_WIDTH);
    }
    CASE(ext_intel_device_info_uuid) {
      return has_info_desc(UR_DEVICE_INFO_UUID);
    }
    CASE(ext_intel_max_mem_bandwidth) {
      // currently not supported
      return false;
    }
    CASE(ext_intel_current_clock_throttle_reasons) {
      return has_info_desc(UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS);
    }
    CASE(ext_intel_fan_speed) {
      return has_info_desc(UR_DEVICE_INFO_FAN_SPEED);
    }
    CASE(ext_intel_power_limits) {
      return has_info_desc(UR_DEVICE_INFO_MIN_POWER_LIMIT) &&
             has_info_desc(UR_DEVICE_INFO_MAX_POWER_LIMIT);
    }
    CASE(ext_oneapi_srgb) { return get_info<info::device::ext_oneapi_srgb>(); }
    CASE(ext_oneapi_native_assert) {
      return get_info_impl<UR_DEVICE_INFO_USE_NATIVE_ASSERT>();
    }
    CASE(ext_oneapi_cuda_async_barrier) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_ASYNC_BARRIER>().value_or(0);
    }
    CASE(ext_intel_legacy_image) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_IMAGE_SUPPORT>().value_or(0);
    }
    CASE(ext_oneapi_bindless_images) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_shared_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_1d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_2d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_external_memory_import) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_external_semaphore_import) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_mipmap) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_mipmap_anisotropy) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_mipmap_level_reference) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_sampled_image_fetch_1d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_sampled_image_fetch_1d) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_sampled_image_fetch_2d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_sampled_image_fetch_2d) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_sampled_image_fetch_3d) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_gather) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_IMAGES_GATHER_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_cubemap) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_cubemap_seamless_filtering) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_image_array) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_unique_addressing_per_dim) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_sample_1d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_bindless_images_sample_2d_usm) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_intel_esimd) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_ESIMD_SUPPORT>().value_or(0);
    }
    CASE(ext_oneapi_ballot_group) {
      return (this->getBackend() == backend::ext_oneapi_level_zero) ||
             (this->getBackend() == backend::opencl) ||
             (this->getBackend() == backend::ext_oneapi_cuda);
    }
    CASE(ext_oneapi_fixed_size_group) {
      return (this->getBackend() == backend::ext_oneapi_level_zero) ||
             (this->getBackend() == backend::opencl) ||
             (this->getBackend() == backend::ext_oneapi_cuda);
    }
    CASE(ext_oneapi_opportunistic_group) {
      return (this->getBackend() == backend::ext_oneapi_level_zero) ||
             (this->getBackend() == backend::opencl) ||
             (this->getBackend() == backend::ext_oneapi_cuda);
    }
    CASE(ext_oneapi_tangle_group) {
      // TODO: tangle_group is not currently supported for CUDA devices. Add
      // when implemented.
      return (this->getBackend() == backend::ext_oneapi_level_zero) ||
             (this->getBackend() == backend::opencl);
    }
    CASE(ext_intel_matrix) {
      using arch = sycl::ext::oneapi::experimental::architecture;
      const arch supported_archs[] = {
          arch::intel_cpu_spr,     arch::intel_cpu_gnr,
          arch::intel_cpu_dmr,     arch::intel_gpu_pvc,
          arch::intel_gpu_dg2_g10, arch::intel_gpu_dg2_g11,
          arch::intel_gpu_dg2_g12, arch::intel_gpu_bmg_g21,
          arch::intel_gpu_lnl_m,   arch::intel_gpu_arl_h,
          arch::intel_gpu_ptl_h,   arch::intel_gpu_ptl_u,
      };
      try {
        return std::any_of(
            std::begin(supported_archs), std::end(supported_archs),
            [=](const arch a) { return this->extOneapiArchitectureIs(a); });
      } catch (const sycl::exception &) {
        // If we're here it means the device does not support architecture
        // querying
        return false;
      }
    }
    CASE(ext_oneapi_is_composite) {
      auto components = CALL_GET_INFO<
          sycl::ext::oneapi::experimental::info::device::component_devices>();
      // Any device with ext_oneapi_is_composite aspect will have at least two
      // constituent component devices.
      return components.size() >= 2;
    }
    CASE(ext_oneapi_is_component) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_COMPOSITE_DEVICE>().value_or(
                 nullptr) != nullptr;
    }
    CASE(ext_oneapi_graph) {
      ur_device_command_buffer_update_capability_flags_t UpdateCapabilities;
      bool CallSuccessful =
          getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
              MDevice, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
              sizeof(UpdateCapabilities), &UpdateCapabilities,
              nullptr) == UR_RESULT_SUCCESS;
      if (!CallSuccessful) {
        return false;
      }

      /* The kernel handle update capability is not yet required for the
       * ext_oneapi_graph aspect */
      ur_device_command_buffer_update_capability_flags_t RequiredCapabilities =
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE;

      return has(aspect::ext_oneapi_limited_graph) &&
             (UpdateCapabilities & RequiredCapabilities) ==
                 RequiredCapabilities;
    }
    CASE(ext_oneapi_limited_graph) {
      bool SupportsCommandBuffers = false;
      bool CallSuccessful =
          getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
              MDevice, UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP,
              sizeof(SupportsCommandBuffers), &SupportsCommandBuffers,
              nullptr) == UR_RESULT_SUCCESS;
      if (!CallSuccessful) {
        return false;
      }

      return SupportsCommandBuffers;
    }
    CASE(ext_oneapi_private_alloca) {
      // Extension only supported on SPIR-V targets.
      backend be = getBackend();
      return be == sycl::backend::ext_oneapi_level_zero ||
             be == sycl::backend::opencl;
    }
    CASE(ext_oneapi_queue_profiling_tag) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP>()
          .value_or(0);
    }
    CASE(ext_oneapi_virtual_mem) {
      return get_info_impl_nocheck<UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT>()
          .value_or(0);
    }
    CASE(ext_intel_fpga_task_sequence) { return is_accelerator(); }
    CASE(ext_oneapi_atomic16) {
      // Likely L0 doesn't check it properly. Need to double-check.
      return has_extension("cl_ext_float_atomics");
    }
    CASE(ext_oneapi_virtual_functions) {
      // TODO: move to UR like e.g. aspect::ext_oneapi_virtual_mem
      backend BE = getBackend();
      bool isCompatibleBE = BE == sycl::backend::ext_oneapi_level_zero ||
                            BE == sycl::backend::opencl;
      return (is_cpu() || is_gpu()) && isCompatibleBE;
    }
    CASE(ext_intel_spill_memory_size) {
      backend BE = getBackend();
      bool isCompatibleBE = BE == sycl::backend::ext_oneapi_level_zero;
      return is_gpu() && isCompatibleBE;
    }
    CASE(ext_oneapi_async_memory_alloc) {
      return get_info_impl_nocheck<
                 UR_DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP>()
          .value_or(0);
    }
    else {
      return false; // This device aspect has not been implemented yet.
    }

#undef CASE
  }

  bool has(aspect Aspect) const {
    switch (Aspect) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case aspect::ASPECT:                                                         \
    return has<aspect::ASPECT>();
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MSG) __SYCL_ASPECT(ASPECT, ID)
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
    }
    assert(false && "Why doesn't has<aspect>() cover it?");
    return false;
  }

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Check if affinity partitioning by specified domain is supported by
  /// device
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of
  /// SYCL Spec
  /// \return true if AffinityDomain is supported by device.
  bool
  is_affinity_supported(info::partition_affinity_domain AffinityDomain) const;

  /// Gets the native handle of the SYCL device.
  ///
  /// \return a native handle.
  ur_native_handle_t getNative() const;

  bool isRootDevice() const { return MRootDevice == nullptr; }

  bool
  extOneapiArchitectureIs(ext::oneapi::experimental::architecture Arch) const {

    return Arch ==
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
           get_info
#else
           get_info_abi_workaround
#endif
           <ext::oneapi::experimental::info::device::architecture>();
  }

  bool extOneapiArchitectureIs(
      ext::oneapi::experimental::arch_category Category) const {
    std::optional<ext::oneapi::experimental::architecture> CategoryMinArch =
        get_category_min_architecture(Category);
    std::optional<ext::oneapi::experimental::architecture> CategoryMaxArch =
        get_category_max_architecture(Category);
    if (CategoryMinArch.has_value() && CategoryMaxArch.has_value()) {
      auto Arch =
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
          get_info
#else
          get_info_abi_workaround
#endif
          <ext::oneapi::experimental::info::device::architecture>();
      return CategoryMinArch <= Arch && Arch <= CategoryMaxArch;
    }
    return false;
  }

  bool extOneapiCanBuild(ext::oneapi::experimental::source_language Language);
  bool extOneapiCanCompile(ext::oneapi::experimental::source_language Language);

  // Returns all guarantees that are either equal to guarantee or weaker than
  // it. E.g if guarantee == parallel, it returns the vector {weakly_parallel,
  // parallel}.
  static std::vector<ext::oneapi::experimental::forward_progress_guarantee>
  getProgressGuaranteesUpTo(
      ext::oneapi::experimental::forward_progress_guarantee guarantee) {
    const int forwardProgressGuaranteeSize = 3;
    int guaranteeVal = static_cast<int>(guarantee);
    std::vector<ext::oneapi::experimental::forward_progress_guarantee> res;
    res.reserve(forwardProgressGuaranteeSize - guaranteeVal);
    for (int currentGuarantee = forwardProgressGuaranteeSize - 1;
         currentGuarantee >= guaranteeVal; --currentGuarantee) {
      res.emplace_back(
          static_cast<ext::oneapi::experimental::forward_progress_guarantee>(
              currentGuarantee));
    }
    return res;
  }

  static sycl::ext::oneapi::experimental::forward_progress_guarantee
  getHostProgressGuarantee(
      sycl::ext::oneapi::experimental::execution_scope threadScope,
      sycl::ext::oneapi::experimental::execution_scope coordinationScope);

  sycl::ext::oneapi::experimental::forward_progress_guarantee
  getProgressGuarantee(
      ext::oneapi::experimental::execution_scope threadScope,
      ext::oneapi::experimental::execution_scope coordinationScope) const;

  bool supportsForwardProgress(
      ext::oneapi::experimental::forward_progress_guarantee guarantee,
      ext::oneapi::experimental::execution_scope threadScope,
      ext::oneapi::experimental::execution_scope coordinationScope) const;

  ext::oneapi::experimental::forward_progress_guarantee
  getImmediateProgressGuarantee(
      ext::oneapi::experimental::execution_scope coordination_scope) const;

  /// Gets the current device timestamp
  /// @throw sycl::feature_not_supported if feature is not supported on device
  uint64_t getCurrentDeviceTime();

  /// Resets the recorded device and host time bases.
  void resetRecordedTimeBases();

  /// Get the backend of this device
  backend getBackend() const { return MPlatform->getBackend(); }

  /// @brief  Get the platform impl serving this device
  platform_impl &getPlatformImpl() const { return *MPlatform; }

  template <ur_device_info_t Desc>
  std::vector<info::fp_config> get_fp_config() const {
    if (Desc == UR_DEVICE_INFO_HALF_FP_CONFIG &&
        !get_info<info::device::native_vector_width_half>())
      return {};
    if (Desc == UR_DEVICE_INFO_DOUBLE_FP_CONFIG &&
        !get_info<info::device::native_vector_width_double>())
      return {};
    auto bits = get_info_impl<Desc>();

    std::vector<info::fp_config> result;
    using cfg = info::fp_config;
    constexpr std::pair<ur_device_fp_capability_flags_t, cfg> mapping[] = {
        {UR_DEVICE_FP_CAPABILITY_FLAG_DENORM, cfg::denorm},
        {UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN, cfg::inf_nan},
        {UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST, cfg::round_to_nearest},
        {UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO, cfg::round_to_zero},
        {UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF, cfg::round_to_inf},
        {UR_DEVICE_FP_CAPABILITY_FLAG_FMA, cfg::fma},
        {UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT, cfg::soft_float},
        {UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT,
         cfg::correctly_rounded_divide_sqrt}};
    for (auto [k, v] : mapping)
      if (bits & k)
        result.push_back(v);
    return result;
  }

  static std::vector<memory_order>
  readMemoryOrderBitfield(ur_memory_order_capability_flags_t bits) {
    std::vector<memory_order> result;
    if (bits & UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED)
      result.push_back(memory_order::relaxed);
    if (bits & UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE)
      result.push_back(memory_order::acquire);
    if (bits & UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE)
      result.push_back(memory_order::release);
    if (bits & UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL)
      result.push_back(memory_order::acq_rel);
    if (bits & UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST)
      result.push_back(memory_order::seq_cst);
    return result;
  }

  static std::vector<memory_scope>
  readMemoryScopeBitfield(ur_memory_scope_capability_flags_t bits) {
    std::vector<memory_scope> result;
    if (bits & UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM)
      result.push_back(memory_scope::work_item);
    if (bits & UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP)
      result.push_back(memory_scope::sub_group);
    if (bits & UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP)
      result.push_back(memory_scope::work_group);
    if (bits & UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE)
      result.push_back(memory_scope::device);
    if (bits & UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM)
      result.push_back(memory_scope::system);
    return result;
  }

  ext::oneapi::experimental::architecture get_architecture() const {
    using oneapi_exp_arch = sycl::ext::oneapi::experimental::architecture;

    // Only for NVIDIA and AMD GPU architectures
    constexpr std::pair<const char *, oneapi_exp_arch>
        NvidiaAmdGPUArchitectures[] = {
            {"5.0", oneapi_exp_arch::nvidia_gpu_sm_50},
            {"5.2", oneapi_exp_arch::nvidia_gpu_sm_52},
            {"5.3", oneapi_exp_arch::nvidia_gpu_sm_53},
            {"6.0", oneapi_exp_arch::nvidia_gpu_sm_60},
            {"6.1", oneapi_exp_arch::nvidia_gpu_sm_61},
            {"6.2", oneapi_exp_arch::nvidia_gpu_sm_62},
            {"7.0", oneapi_exp_arch::nvidia_gpu_sm_70},
            {"7.2", oneapi_exp_arch::nvidia_gpu_sm_72},
            {"7.5", oneapi_exp_arch::nvidia_gpu_sm_75},
            {"8.0", oneapi_exp_arch::nvidia_gpu_sm_80},
            {"8.6", oneapi_exp_arch::nvidia_gpu_sm_86},
            {"8.7", oneapi_exp_arch::nvidia_gpu_sm_87},
            {"8.9", oneapi_exp_arch::nvidia_gpu_sm_89},
            {"9.0", oneapi_exp_arch::nvidia_gpu_sm_90},
            {"gfx701", oneapi_exp_arch::amd_gpu_gfx701},
            {"gfx702", oneapi_exp_arch::amd_gpu_gfx702},
            {"gfx703", oneapi_exp_arch::amd_gpu_gfx703},
            {"gfx704", oneapi_exp_arch::amd_gpu_gfx704},
            {"gfx705", oneapi_exp_arch::amd_gpu_gfx705},
            {"gfx801", oneapi_exp_arch::amd_gpu_gfx801},
            {"gfx802", oneapi_exp_arch::amd_gpu_gfx802},
            {"gfx803", oneapi_exp_arch::amd_gpu_gfx803},
            {"gfx805", oneapi_exp_arch::amd_gpu_gfx805},
            {"gfx810", oneapi_exp_arch::amd_gpu_gfx810},
            {"gfx900", oneapi_exp_arch::amd_gpu_gfx900},
            {"gfx902", oneapi_exp_arch::amd_gpu_gfx902},
            {"gfx904", oneapi_exp_arch::amd_gpu_gfx904},
            {"gfx906", oneapi_exp_arch::amd_gpu_gfx906},
            {"gfx908", oneapi_exp_arch::amd_gpu_gfx908},
            {"gfx909", oneapi_exp_arch::amd_gpu_gfx909},
            {"gfx90a", oneapi_exp_arch::amd_gpu_gfx90a},
            {"gfx90c", oneapi_exp_arch::amd_gpu_gfx90c},
            {"gfx940", oneapi_exp_arch::amd_gpu_gfx940},
            {"gfx941", oneapi_exp_arch::amd_gpu_gfx941},
            {"gfx942", oneapi_exp_arch::amd_gpu_gfx942},
            {"gfx1010", oneapi_exp_arch::amd_gpu_gfx1010},
            {"gfx1011", oneapi_exp_arch::amd_gpu_gfx1011},
            {"gfx1012", oneapi_exp_arch::amd_gpu_gfx1012},
            {"gfx1013", oneapi_exp_arch::amd_gpu_gfx1013},
            {"gfx1030", oneapi_exp_arch::amd_gpu_gfx1030},
            {"gfx1031", oneapi_exp_arch::amd_gpu_gfx1031},
            {"gfx1032", oneapi_exp_arch::amd_gpu_gfx1032},
            {"gfx1033", oneapi_exp_arch::amd_gpu_gfx1033},
            {"gfx1034", oneapi_exp_arch::amd_gpu_gfx1034},
            {"gfx1035", oneapi_exp_arch::amd_gpu_gfx1035},
            {"gfx1036", oneapi_exp_arch::amd_gpu_gfx1036},
            {"gfx1100", oneapi_exp_arch::amd_gpu_gfx1100},
            {"gfx1101", oneapi_exp_arch::amd_gpu_gfx1101},
            {"gfx1102", oneapi_exp_arch::amd_gpu_gfx1102},
            {"gfx1103", oneapi_exp_arch::amd_gpu_gfx1103},
            {"gfx1150", oneapi_exp_arch::amd_gpu_gfx1150},
            {"gfx1151", oneapi_exp_arch::amd_gpu_gfx1151},
            {"gfx1200", oneapi_exp_arch::amd_gpu_gfx1200},
            {"gfx1201", oneapi_exp_arch::amd_gpu_gfx1201},
        };

    // Only for Intel GPU architectures
    constexpr std::pair<const int, oneapi_exp_arch> IntelGPUArchitectures[] = {
        {0x02000000, oneapi_exp_arch::intel_gpu_bdw},
        {0x02400009, oneapi_exp_arch::intel_gpu_skl},
        {0x02404009, oneapi_exp_arch::intel_gpu_kbl},
        {0x02408009, oneapi_exp_arch::intel_gpu_cfl},
        {0x0240c000, oneapi_exp_arch::intel_gpu_apl},
        {0x02410000, oneapi_exp_arch::intel_gpu_glk},
        {0x02414000, oneapi_exp_arch::intel_gpu_whl},
        {0x02418000, oneapi_exp_arch::intel_gpu_aml},
        {0x0241c000, oneapi_exp_arch::intel_gpu_cml},
        {0x02c00000, oneapi_exp_arch::intel_gpu_icllp},
        {0x02c08000, oneapi_exp_arch::intel_gpu_ehl},
        {0x03000000, oneapi_exp_arch::intel_gpu_tgllp},
        {0x03004000, oneapi_exp_arch::intel_gpu_rkl},
        {0x03008000, oneapi_exp_arch::intel_gpu_adl_s},
        {0x0300c000, oneapi_exp_arch::intel_gpu_adl_p},
        {0x03010000, oneapi_exp_arch::intel_gpu_adl_n},
        {0x03028000, oneapi_exp_arch::intel_gpu_dg1},
        {0x030dc000, oneapi_exp_arch::intel_gpu_acm_g10}, // A0
        {0x030dc001, oneapi_exp_arch::intel_gpu_acm_g10}, // A1
        {0x030dc004, oneapi_exp_arch::intel_gpu_acm_g10}, // B0
        {0x030dc008, oneapi_exp_arch::intel_gpu_acm_g10}, // C0
        {0x030e0000, oneapi_exp_arch::intel_gpu_acm_g11}, // A0
        {0x030e0004, oneapi_exp_arch::intel_gpu_acm_g11}, // B0
        {0x030e0005, oneapi_exp_arch::intel_gpu_acm_g11}, // B1
        {0x030e4000, oneapi_exp_arch::intel_gpu_acm_g12}, // A0
        {0x030f0000, oneapi_exp_arch::intel_gpu_pvc},     // XL-A0
        {0x030f0001, oneapi_exp_arch::intel_gpu_pvc},     // XL-AOP
        {0x030f0003, oneapi_exp_arch::intel_gpu_pvc},     // XT-A0
        {0x030f0005, oneapi_exp_arch::intel_gpu_pvc},     // XT-B0
        {0x030f0006, oneapi_exp_arch::intel_gpu_pvc},     // XT-B1
        {0x030f0007, oneapi_exp_arch::intel_gpu_pvc},     // XT-C0
        {0x030f4007, oneapi_exp_arch::intel_gpu_pvc_vg},  // C0
        {0x03118000, oneapi_exp_arch::intel_gpu_mtl_u},   // A0
        {0x03118004, oneapi_exp_arch::intel_gpu_mtl_u},   // B0
        {0x0311c000, oneapi_exp_arch::intel_gpu_mtl_h},   // A0
        {0x0311c004, oneapi_exp_arch::intel_gpu_mtl_h},   // B0
        {0x03128000, oneapi_exp_arch::intel_gpu_arl_h},   // A0
        {0x03128004, oneapi_exp_arch::intel_gpu_arl_h},   // B0
        {0x05004000, oneapi_exp_arch::intel_gpu_bmg_g21}, // A0
        {0x05004001, oneapi_exp_arch::intel_gpu_bmg_g21}, // A1
        {0x05004004, oneapi_exp_arch::intel_gpu_bmg_g21}, // B0
        {0x05010000, oneapi_exp_arch::intel_gpu_lnl_m},   // A0
        {0x05010001, oneapi_exp_arch::intel_gpu_lnl_m},   // A1
        {0x05010004, oneapi_exp_arch::intel_gpu_lnl_m},   // B0
        {0x07800000, oneapi_exp_arch::intel_gpu_ptl_h},   // A0
        {0x07800004, oneapi_exp_arch::intel_gpu_ptl_h},   // B0
        {0x07804000, oneapi_exp_arch::intel_gpu_ptl_u},   // A0
        {0x07804001, oneapi_exp_arch::intel_gpu_ptl_u},   // A1
    };

    // Only for Intel CPU architectures
    constexpr std::pair<const int, oneapi_exp_arch> IntelCPUArchitectures[] = {
        {8, oneapi_exp_arch::intel_cpu_spr},
        {9, oneapi_exp_arch::intel_cpu_gnr},
        {10, oneapi_exp_arch::intel_cpu_dmr},
    };
    backend CurrentBackend = getBackend();
    auto LookupIPVersion = [&, this](auto &ArchList)
        -> std::optional<ext::oneapi::experimental::architecture> {
      auto DeviceIp = get_info_impl_nocheck<UR_DEVICE_INFO_IP_VERSION>();
      if (!DeviceIp.has_val()) {
        ur_result_t Err = DeviceIp.error();
        if (Err == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
          // Not all devices support this device info query
          return std::nullopt;
        }
        getAdapter()->checkUrResult(Err);
      }

      auto Val = static_cast<int>(DeviceIp.value());
      for (const auto &Item : ArchList) {
        if (Item.first == Val)
          return Item.second;
      }
      return std::nullopt;
    };

    if (is_gpu() && (backend::ext_oneapi_level_zero == CurrentBackend ||
                     backend::opencl == CurrentBackend)) {
      return LookupIPVersion(IntelGPUArchitectures)
          .value_or(ext::oneapi::experimental::architecture::unknown);
    } else if (is_gpu() && (backend::ext_oneapi_cuda == CurrentBackend ||
                            backend::ext_oneapi_hip == CurrentBackend)) {
      auto MapArchIDToArchName = [&](const char *arch) {
        for (const auto &Item : NvidiaAmdGPUArchitectures) {
          if (std::string_view(Item.first) == arch)
            return Item.second;
        }
        return ext::oneapi::experimental::architecture::unknown;
      };
      std::string DeviceArch =
          get_info_impl<UrInfoCode<info::device::version>::value>();
      std::string_view DeviceArchSubstr =
          std::string_view{DeviceArch}.substr(0, DeviceArch.find(":"));
      return MapArchIDToArchName(DeviceArchSubstr.data());
    } else if (is_cpu() && backend::opencl == CurrentBackend) {
      return LookupIPVersion(IntelCPUArchitectures)
          .value_or(ext::oneapi::experimental::architecture::x86_64);
    } // else is not needed
    // TODO: add support of other architectures by extending with else if
    return ext::oneapi::experimental::architecture::unknown;
  }

  std::vector<ext::oneapi::experimental::matrix::combination>
  get_matrix_combinations() const {
    using namespace ext::oneapi::experimental::matrix;
    using namespace ext::oneapi::experimental;
    backend CurrentBackend = getBackend();
    auto get_current_architecture = [this]() -> std::optional<architecture> {
      // this helper lambda ignores all runtime-related exceptions from
      // quering the device architecture. For instance, if device architecture
      // on user's machine is not supported by
      // sycl_ext_oneapi_device_architecture, the runtime exception is
      // omitted, and std::nullopt is returned.
      try {
        return CALL_GET_INFO<
            ext::oneapi::experimental::info::device::architecture>();
      } catch (sycl::exception &e) {
        if (e.code() != errc::runtime)
          std::rethrow_exception(std::make_exception_ptr(e));
      }
      return std::nullopt;
    };
    std::optional<architecture> DeviceArchOpt = get_current_architecture();
    if (!DeviceArchOpt.has_value())
      return {};
    architecture DeviceArch = DeviceArchOpt.value();
    if (architecture::intel_cpu_spr == DeviceArch)
      return {
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::intel_cpu_gnr == DeviceArch)
      return {
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {16, 16, 32, 0, 0, 0, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::intel_cpu_dmr == DeviceArch)
      return {
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {16, 16, 32, 0, 0, 0, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {16, 16, 16, 0, 0, 0, matrix_type::tf32, matrix_type::tf32,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if ((architecture::intel_gpu_pvc == DeviceArch) ||
             (architecture::intel_gpu_bmg_g21 == DeviceArch) ||
             (architecture::intel_gpu_lnl_m == DeviceArch) ||
             (architecture::intel_gpu_ptl_h == DeviceArch) ||
             (architecture::intel_gpu_ptl_u == DeviceArch)) {
      std::vector<ext::oneapi::experimental::matrix::combination> pvc_combs = {
          {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 1, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 1, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 1, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 1, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 32, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 32, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 32, 64, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 1, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 1, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 1, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 1, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 32, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 32, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 32, 64, 32, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {0, 0, 0, 1, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 1, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {0, 0, 0, 1, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {0, 0, 0, 1, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {0, 0, 0, 32, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::fp32},
          {0, 0, 0, 32, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::bf16},
          {0, 0, 0, 32, 64, 32, matrix_type::bf16, matrix_type::bf16,
           matrix_type::bf16, matrix_type::bf16},
          {8, 0, 0, 0, 16, 8, matrix_type::tf32, matrix_type::tf32,
           matrix_type::fp32, matrix_type::fp32},
      };
      return pvc_combs;
    } else if ((architecture::intel_gpu_dg2_g10 == DeviceArch) ||
               (architecture::intel_gpu_dg2_g11 == DeviceArch) ||
               (architecture::intel_gpu_dg2_g12 == DeviceArch) ||
               (architecture::intel_gpu_arl_h == DeviceArch))
      return {
          {8, 0, 0, 0, 8, 32, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::uint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::sint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 32, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {8, 0, 0, 0, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {8, 0, 0, 0, 8, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 32, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
      };
    else if (architecture::amd_gpu_gfx90a == DeviceArch)
      return {
          {0, 0, 0, 32, 32, 8, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 32, 8, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 32, 8, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 4, matrix_type::fp64, matrix_type::fp64,
           matrix_type::fp64, matrix_type::fp64},
      };
    else if (backend::ext_oneapi_cuda == CurrentBackend) {
      // TODO: Tho following can be simplified when comparison of
      // architectures using < and > will be implemented
      using oneapi_exp_arch = sycl::ext::oneapi::experimental::architecture;
      constexpr std::pair<float, oneapi_exp_arch> NvidiaArchNumbs[] = {
          {5.0, oneapi_exp_arch::nvidia_gpu_sm_50},
          {5.2, oneapi_exp_arch::nvidia_gpu_sm_52},
          {5.3, oneapi_exp_arch::nvidia_gpu_sm_53},
          {6.0, oneapi_exp_arch::nvidia_gpu_sm_60},
          {6.1, oneapi_exp_arch::nvidia_gpu_sm_61},
          {6.2, oneapi_exp_arch::nvidia_gpu_sm_62},
          {7.0, oneapi_exp_arch::nvidia_gpu_sm_70},
          {7.2, oneapi_exp_arch::nvidia_gpu_sm_72},
          {7.5, oneapi_exp_arch::nvidia_gpu_sm_75},
          {8.0, oneapi_exp_arch::nvidia_gpu_sm_80},
          {8.6, oneapi_exp_arch::nvidia_gpu_sm_86},
          {8.7, oneapi_exp_arch::nvidia_gpu_sm_87},
          {8.9, oneapi_exp_arch::nvidia_gpu_sm_89},
          {9.0, oneapi_exp_arch::nvidia_gpu_sm_90},
      };
      auto GetArchNum = [&](const architecture &arch) {
        for (const auto &Item : NvidiaArchNumbs)
          if (Item.second == arch)
            return Item.first;
        return 0.f;
      };
      float ComputeCapability = GetArchNum(DeviceArch);
      std::vector<combination> sm_70_combinations = {
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp32, matrix_type::fp16},
          {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
           matrix_type::fp16, matrix_type::fp32}};
      std::vector<combination> sm_72_combinations = {
          {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 8, 32, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 8, 16, matrix_type::sint8, matrix_type::sint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 16, 16, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 8, 32, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32},
          {0, 0, 0, 32, 8, 16, matrix_type::uint8, matrix_type::uint8,
           matrix_type::sint32, matrix_type::sint32}};
      std::vector<combination> sm_80_combinations = {
          {0, 0, 0, 16, 16, 8, matrix_type::tf32, matrix_type::tf32,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 32, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 32, 8, 16, matrix_type::bf16, matrix_type::bf16,
           matrix_type::fp32, matrix_type::fp32},
          {0, 0, 0, 8, 8, 4, matrix_type::fp64, matrix_type::fp64,
           matrix_type::fp64, matrix_type::fp64}};
      if (ComputeCapability >= 8.0) {
        sm_80_combinations.insert(sm_80_combinations.end(),
                                  sm_72_combinations.begin(),
                                  sm_72_combinations.end());
        sm_80_combinations.insert(sm_80_combinations.end(),
                                  sm_70_combinations.begin(),
                                  sm_70_combinations.end());
        return sm_80_combinations;
      } else if (ComputeCapability >= 7.2) {
        sm_72_combinations.insert(sm_72_combinations.end(),
                                  sm_70_combinations.begin(),
                                  sm_70_combinations.end());
        return sm_72_combinations;
      } else if (ComputeCapability >= 7.0)
        return sm_70_combinations;
    }
    return {};
  }

private:
  ur_device_handle_t MDevice = 0;
  // This is used for getAdapter so should be above other properties.
  std::shared_ptr<platform_impl> MPlatform;

  std::shared_mutex MDeviceHostBaseTimeMutex;
  std::pair<uint64_t, uint64_t> MDeviceHostBaseTime{0, 0};

  const ur_device_handle_t MRootDevice;

  // Order of caches matters! UR must come before SYCL info descriptors (because
  // get_info calls get_info_impl but the opposite never happens) and both
  // should come before aspects.
  //
  // To make an addition property cacheable just expand one of the caches below
  // with that property, no other changes should be necessary.
  mutable JointCache<
      UREagerCache<UR_DEVICE_INFO_TYPE, UR_DEVICE_INFO_USE_NATIVE_ASSERT,
                   UR_DEVICE_INFO_EXTENSIONS>, //
      URCallOnceCache<UR_DEVICE_INFO_NAME,
                      // USM:
                      UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
                      UR_DEVICE_INFO_USM_HOST_SUPPORT,
                      UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
                      UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
                      UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
                      //
                      UR_DEVICE_INFO_ATOMIC_64>, //
      EagerCache<InfoInitializer>,               //
      CallOnceCache<InfoInitializer,
                    ext::oneapi::experimental::info::device::architecture>, //
      AspectCache<EagerCache, aspect::fp16, aspect::fp64,
                  aspect::int64_base_atomics, aspect::int64_extended_atomics,
                  aspect::ext_oneapi_atomic16>,
      AspectCache<
          CallOnceCache,
          // Slow, >100ns (for baseline cached ~30..40ns):
          aspect::ext_intel_pci_address, aspect::ext_intel_gpu_eu_count,
          aspect::ext_intel_free_memory, aspect::ext_intel_fan_speed,
          aspect::ext_intel_power_limits,
          // medium-slow, 60-90ns (for baseline cached ~30..40ns):
          aspect::ext_intel_gpu_eu_simd_width, aspect::ext_intel_gpu_slices,
          aspect::ext_intel_gpu_subslices_per_slice,
          aspect::ext_intel_gpu_eu_count_per_subslice,
          aspect::ext_intel_device_info_uuid,
          aspect::ext_intel_gpu_hw_threads_per_eu,
          aspect::ext_intel_memory_clock_rate,
          aspect::ext_intel_memory_bus_width,
          aspect::ext_oneapi_bindless_images,
          aspect::ext_oneapi_bindless_images_1d_usm,
          aspect::ext_oneapi_bindless_images_2d_usm,
          aspect::ext_oneapi_is_composite, aspect::ext_oneapi_is_component>>
      MCache;

}; // class device_impl

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <typename Param>
typename Param::return_type device_impl::get_info() const {
  return get_info_abi_workaround<Param>();
}

#define EXPORT_GET_INFO(PARAM)                                                 \
  template <>                                                                  \
  __SYCL_EXPORT PARAM::return_type device_impl::get_info<PARAM>() const;

// clang-format off
EXPORT_GET_INFO(ext::intel::info::device::device_id)
EXPORT_GET_INFO(ext::intel::info::device::pci_address)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_simd_width)
EXPORT_GET_INFO(ext::intel::info::device::gpu_slices)
EXPORT_GET_INFO(ext::intel::info::device::gpu_subslices_per_slice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count_per_subslice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_hw_threads_per_eu)
EXPORT_GET_INFO(ext::intel::info::device::max_mem_bandwidth)
EXPORT_GET_INFO(ext::intel::info::device::uuid)
EXPORT_GET_INFO(ext::intel::info::device::free_memory)
EXPORT_GET_INFO(ext::intel::info::device::memory_clock_rate)
EXPORT_GET_INFO(ext::intel::info::device::memory_bus_width)
EXPORT_GET_INFO(ext::intel::info::device::max_compute_queue_indices)
EXPORT_GET_INFO(ext::intel::esimd::info::device::has_2d_block_io_support)
EXPORT_GET_INFO(ext::intel::info::device::current_clock_throttle_reasons)
EXPORT_GET_INFO(ext::intel::info::device::fan_speed)
EXPORT_GET_INFO(ext::intel::info::device::min_power_limit)
EXPORT_GET_INFO(ext::intel::info::device::max_power_limit)

EXPORT_GET_INFO(ext::codeplay::experimental::info::device::supports_fusion)
EXPORT_GET_INFO(ext::codeplay::experimental::info::device::max_registers_per_work_group)

EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_global_work_groups)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<1>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<2>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<3>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::sub_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::architecture)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::matrix_combinations)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::image_row_pitch_align)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_row_pitch)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_width)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_height)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::mipmap_max_anisotropy)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::component_devices)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::composite_device)
EXPORT_GET_INFO(ext::oneapi::info::device::num_compute_units)
// clang-format on

#undef EXPORT_GET_INFO
#endif

#undef CALL_GET_INFO
} // namespace detail
} // namespace _V1
} // namespace sycl
