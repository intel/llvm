//==------ enqueue_functions.hpp ------- SYCL enqueue free functions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility> // for std::forward

#include <sycl/event.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/handler.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {
// Trait for identifying sycl::range and sycl::nd_range.
template <typename RangeT> struct is_range_or_nd_range : std::false_type {};
template <int Dimensions>
struct is_range_or_nd_range<range<Dimensions>> : std::true_type {};
template <int Dimensions>
struct is_range_or_nd_range<nd_range<Dimensions>> : std::true_type {};

template <typename RangeT>
constexpr bool is_range_or_nd_range_v = is_range_or_nd_range<RangeT>::value;

template <typename LCRangeT, typename LCPropertiesT> struct LaunchConfigAccess;
} // namespace detail

// Available only when Range is range or nd_range
template <
    typename RangeT, typename PropertiesT = empty_properties_t,
    typename = std::enable_if_t<
        ext::oneapi::experimental::detail::is_range_or_nd_range_v<RangeT>>>
class launch_config {
public:
  launch_config(RangeT Range, PropertiesT Properties = {})
      : MRange{Range}, MProperties{Properties} {}

private:
  RangeT MRange;
  PropertiesT MProperties;

  const RangeT &getRange() const noexcept { return MRange; }

  const PropertiesT &getProperties() const noexcept { return MProperties; }

  template <typename LCRangeT, typename LCPropertiesT>
  friend struct detail::LaunchConfigAccess;
};

namespace detail {
// Helper for accessing the members of launch_config.
template <typename LCRangeT, typename LCPropertiesT> struct LaunchConfigAccess {
  LaunchConfigAccess(const launch_config<LCRangeT, LCPropertiesT> &LaunchConfig)
      : MLaunchConfig{LaunchConfig} {}

  const launch_config<LCRangeT, LCPropertiesT> &MLaunchConfig;

  const LCRangeT &getRange() const noexcept { return MLaunchConfig.getRange(); }

  const LCPropertiesT &getProperties() const noexcept {
    return MLaunchConfig.getProperties();
  }
};
} // namespace detail

template <typename CommandGroupFunc>
void submit(queue Q, CommandGroupFunc &&CGF) {
  // TODO: Use new submit without Events.
  Q.submit(std::forward<CommandGroupFunc>(CGF));
}

template <typename CommandGroupFunc>
event submit_with_event(queue Q, CommandGroupFunc &&CGF) {
  return Q.submit(std::forward<CommandGroupFunc>(CGF));
}

template <typename KernelName = sycl::detail::auto_name, typename KernelType>
void single_task(handler &CGH, const KernelType &KernelObj) {
  CGH.single_task<KernelName>(KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, typename KernelType>
void single_task(queue Q, const KernelType &KernelObj) {
  submit(Q, [&](handler &CGH) { single_task<KernelName>(CGH, KernelObj); });
}

template <typename... ArgsT>
void single_task(handler &CGH, const kernel &KernelObj, ArgsT &&...Args) {
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.single_task(KernelObj);
}

template <typename... ArgsT>
void single_task(queue Q, const kernel &KernelObj, ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    single_task(CGH, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

// TODO: Make overloads for scalar arguments for range.
template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename KernelType, typename... ReductionsT>
void parallel_for(handler &CGH, range<Dimensions> Range,
                  const KernelType &KernelObj, ReductionsT &&...Reductions) {
  CGH.parallel_for<KernelName>(Range, std::forward<ReductionsT>(Reductions)...,
                               KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename KernelType, typename... ReductionsT>
void parallel_for(queue Q, range<Dimensions> Range, const KernelType &KernelObj,
                  ReductionsT &&...Reductions) {
  submit(Q, [&](handler &CGH) {
    parallel_for<KernelName>(CGH, Range, KernelObj,
                             std::forward<ReductionsT>(Reductions)...);
  });
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void parallel_for(handler &CGH,
                  launch_config<range<Dimensions>, Properties> Config,
                  const KernelType &KernelObj, ReductionsT &&...Reductions) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.parallel_for<KernelName>(ConfigAccess.getRange(),
                               std::forward<ReductionsT>(Reductions)...,
                               KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void parallel_for(queue Q, launch_config<range<Dimensions>, Properties> Config,
                  const KernelType &KernelObj, ReductionsT &&...Reductions) {
  submit(Q, [&](handler &CGH) {
    parallel_for<KernelName>(CGH, Config, KernelObj,
                             std::forward<ReductionsT>(Reductions)...);
  });
}

template <int Dimensions, typename... ArgsT>
void parallel_for(handler &CGH, range<Dimensions> Range,
                  const kernel &KernelObj, ArgsT &&...Args) {
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.parallel_for(Range, KernelObj);
}

template <int Dimensions, typename... ArgsT>
void parallel_for(queue Q, range<Dimensions> Range, const kernel &KernelObj,
                  ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    parallel_for(CGH, Range, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

template <int Dimensions, typename Properties, typename... ArgsT>
void parallel_for(handler &CGH,
                  launch_config<range<Dimensions>, Properties> Config,
                  const kernel &KernelObj, ArgsT &&...Args) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.parallel_for(ConfigAccess.getRange(), KernelObj);
}

template <int Dimensions, typename Properties, typename... ArgsT>
void parallel_for(queue Q, launch_config<range<Dimensions>, Properties> Config,
                  const kernel &KernelObj, ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    parallel_for(CGH, Config, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename KernelType, typename... ReductionsT>
void nd_launch(handler &CGH, nd_range<Dimensions> Range,
               const KernelType &KernelObj, ReductionsT &&...Reductions) {
  CGH.parallel_for<KernelName>(Range, std::forward<ReductionsT>(Reductions)...,
                               KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename KernelType, typename... ReductionsT>
void nd_launch(queue Q, nd_range<Dimensions> Range, const KernelType &KernelObj,
               ReductionsT &&...Reductions) {
  submit(Q, [&](handler &CGH) {
    nd_launch(CGH, Range, KernelObj, std::forward<ReductionsT>(Reductions)...);
  });
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void nd_launch(handler &CGH,
               launch_config<nd_range<Dimensions>, Properties> Config,
               const KernelType &KernelObj, ReductionsT &&...Reductions) {

  ext::oneapi::experimental::detail::LaunchConfigAccess<nd_range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.parallel_for<KernelName>(ConfigAccess.getRange(),
                               std::forward<ReductionsT>(Reductions)...,
                               KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void nd_launch(queue Q, launch_config<nd_range<Dimensions>, Properties> Config,
               const KernelType &KernelObj, ReductionsT &&...Reductions) {
  submit(Q, [&](handler &CGH) {
    nd_launch(CGH, Config, KernelObj, std::forward<ReductionsT>(Reductions)...);
  });
}

template <int Dimensions, typename... ArgsT>
void nd_launch(handler &CGH, nd_range<Dimensions> Range,
               const kernel &KernelObj, ArgsT &&...Args) {
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.parallel_for(Range, KernelObj);
}

template <int Dimensions, typename... ArgsT>
void nd_launch(queue Q, nd_range<Dimensions> Range, const kernel &KernelObj,
               ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    nd_launch(CGH, Range, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

template <int Dimensions, typename Properties, typename... ArgsT>
void nd_launch(handler &CGH,
               launch_config<nd_range<Dimensions>, Properties> Config,
               const kernel &KernelObj, ArgsT &&...Args) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<nd_range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.parallel_for(ConfigAccess.getRange(), KernelObj);
}

template <int Dimensions, typename Properties, typename... ArgsT>
void nd_launch(queue Q, launch_config<nd_range<Dimensions>, Properties> Config,
               const kernel &KernelObj, ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    nd_launch(CGH, Config, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

inline void memcpy(handler &CGH, void *Dest, const void *Src, size_t NumBytes) {
  CGH.memcpy(Dest, Src, NumBytes);
}

inline void memcpy(queue Q, void *Dest, const void *Src, size_t NumBytes) {
  submit(Q, [&](handler &CGH) { memcpy(CGH, Dest, Src, NumBytes); });
}

template <typename T>
void copy(handler &CGH, const T *Src, T *Dest, size_t Count) {
  CGH.copy<T>(Src, Dest, Count);
}

template <typename T> void copy(queue Q, const T *Src, T *Dest, size_t Count) {
  submit(Q, [&](handler &CGH) { copy<T>(CGH, Src, Dest, Count); });
}

inline void memset(handler &CGH, void *Ptr, int Value, size_t NumBytes) {
  CGH.memset(Ptr, Value, NumBytes);
}

inline void memset(queue Q, void *Ptr, int Value, size_t NumBytes) {
  submit(Q, [&](handler &CGH) { memset(CGH, Ptr, Value, NumBytes); });
}

template <typename T>
void fill(sycl::handler &CGH, T *Ptr, const T &Pattern, size_t Count) {
  CGH.fill(Ptr, Pattern, Count);
}

template <typename T>
void fill(sycl::queue Q, T *Ptr, const T &Pattern, size_t Count) {
  submit(Q, [&](handler &CGH) { fill<T>(CGH, Ptr, Pattern, Count); });
}

inline void prefetch(handler &CGH, void *Ptr, size_t NumBytes) {
  CGH.prefetch(Ptr, NumBytes);
}

inline void prefetch(queue Q, void *Ptr, size_t NumBytes) {
  submit(Q, [&](handler &CGH) { prefetch(CGH, Ptr, NumBytes); });
}

inline void mem_advise(handler &CGH, void *Ptr, size_t NumBytes, int Advice) {
  CGH.mem_advise(Ptr, NumBytes, Advice);
}

inline void mem_advise(queue Q, void *Ptr, size_t NumBytes, int Advice) {
  submit(Q, [&](handler &CGH) { mem_advise(CGH, Ptr, NumBytes, Advice); });
}

inline void barrier(handler &CGH) { CGH.ext_oneapi_barrier(); }

inline void barrier(queue Q) {
  submit(Q, [&](handler &CGH) { barrier(CGH); });
}

inline void partial_barrier(handler &CGH, const std::vector<event> &Events) {
  CGH.ext_oneapi_barrier(Events);
}

inline void partial_barrier(queue Q, const std::vector<event> &Events) {
  submit(Q, [&](handler &CGH) { partial_barrier(CGH, Events); });
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
