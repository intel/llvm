#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {
namespace khr {
namespace detail {
template <typename CommandGroupFunc, typename PropertiesT>
void submit(queue Q, PropertiesT Props, CommandGroupFunc &&CGF,
            const sycl::detail::code_location &CodeLoc =
                sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::detail::submit_impl(
      Q, Props, std::forward<CommandGroupFunc>(CGF), CodeLoc);
}

template <typename CommandGroupFunc, typename PropertiesT>
event submit_with_event(queue Q, PropertiesT Props, CommandGroupFunc &&CGF,
                        const sycl::detail::code_location &CodeLoc =
                            sycl::detail::code_location::current()) {
  return sycl::ext::oneapi::experimental::detail::submit_with_event_impl(
      Q, Props, std::forward<CommandGroupFunc>(CGF), CodeLoc);
}
} // namespace detail

template <typename CommandGroupFunc>
void submit(queue Q, CommandGroupFunc &&CGF,
            const sycl::detail::code_location &CodeLoc =
                sycl::detail::code_location::current()) {
  detail::submit(Q, ext::oneapi::experimental::empty_properties_t{},
                 std::forward<CommandGroupFunc>(CGF), CodeLoc);
}

template <typename CommandGroupFunc>
event submit_with_event(queue Q, CommandGroupFunc &&CGF,
                        const sycl::detail::code_location &CodeLoc =
                            sycl::detail::code_location::current()) {
  return detail::submit_with_event(
      Q, ext::oneapi::experimental::empty_properties_t{},
      std::forward<CommandGroupFunc>(CGF), CodeLoc);
}

template <typename KernelName = sycl::detail::auto_name, typename KernelType>
void launch_task(handler &CGH, const KernelType &KernelObj) {
  CGH.single_task<KernelName>(KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, typename KernelType>
void launch_task(queue Q, const KernelType &KernelObj,
                 const sycl::detail::code_location &CodeLoc =
                     sycl::detail::code_location::current()) {
  submit(
      Q, [&](handler &CGH) { launch_task<KernelName>(CGH, KernelObj); },
      CodeLoc);
}

template <typename... ArgsT>
void launch_task(handler &CGH, const kernel &KernelObj, ArgsT &&...Args) {
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  CGH.single_task(KernelObj);
}

template <typename... ArgsT>
void launch_task(queue Q, const kernel &KernelObj, ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    launch_task(CGH, KernelObj, std::forward<ArgsT>(Args)...);
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
void parallel_for(
    handler &CGH,
    ext::oneapi::experimental::launch_config<range<Dimensions>, Properties>
        Config,
    const KernelType &KernelObj, ReductionsT &&...Reductions) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.parallel_for<KernelName>(
      ConfigAccess.getRange(), ConfigAccess.getProperties(),
      std::forward<ReductionsT>(Reductions)..., KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void parallel_for(
    queue Q,
    ext::oneapi::experimental::launch_config<range<Dimensions>, Properties>
        Config,
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
void parallel_for(
    handler &CGH,
    ext::oneapi::experimental::launch_config<range<Dimensions>, Properties>
        Config,
    const kernel &KernelObj, ArgsT &&...Args) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  sycl::detail::HandlerAccess::parallelForImpl(
      CGH, ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelObj);
}

template <int Dimensions, typename Properties, typename... ArgsT>
void parallel_for(
    queue Q,
    ext::oneapi::experimental::launch_config<range<Dimensions>, Properties>
        Config,
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
    nd_launch<KernelName>(CGH, Range, KernelObj,
                          std::forward<ReductionsT>(Reductions)...);
  });
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void nd_launch(
    handler &CGH,
    ext::oneapi::experimental::launch_config<nd_range<Dimensions>, Properties>
        Config,
    const KernelType &KernelObj, ReductionsT &&...Reductions) {

  ext::oneapi::experimental::detail::LaunchConfigAccess<nd_range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.parallel_for<KernelName>(
      ConfigAccess.getRange(), ConfigAccess.getProperties(),
      std::forward<ReductionsT>(Reductions)..., KernelObj);
}

template <typename KernelName = sycl::detail::auto_name, int Dimensions,
          typename Properties, typename KernelType, typename... ReductionsT>
void nd_launch(
    queue Q,
    ext::oneapi::experimental::launch_config<nd_range<Dimensions>, Properties>
        Config,
    const KernelType &KernelObj, ReductionsT &&...Reductions) {
  submit(Q, [&](handler &CGH) {
    nd_launch<KernelName>(CGH, Config, KernelObj,
                          std::forward<ReductionsT>(Reductions)...);
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
void nd_launch(
    handler &CGH,
    ext::oneapi::experimental::launch_config<nd_range<Dimensions>, Properties>
        Config,
    const kernel &KernelObj, ArgsT &&...Args) {
  ext::oneapi::experimental::detail::LaunchConfigAccess<nd_range<Dimensions>,
                                                        Properties>
      ConfigAccess(Config);
  CGH.set_args<ArgsT...>(std::forward<ArgsT>(Args)...);
  sycl::detail::HandlerAccess::parallelForImpl(
      CGH, ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelObj);
}

template <int Dimensions, typename Properties, typename... ArgsT>
void nd_launch(
    queue Q,
    ext::oneapi::experimental::launch_config<nd_range<Dimensions>, Properties>
        Config,
    const kernel &KernelObj, ArgsT &&...Args) {
  submit(Q, [&](handler &CGH) {
    nd_launch(CGH, Config, KernelObj, std::forward<ArgsT>(Args)...);
  });
}

inline void memcpy(handler &CGH, void *Dest, const void *Src, size_t NumBytes) {
  CGH.memcpy(Dest, Src, NumBytes);
}

__SYCL_EXPORT void memcpy(queue Q, void *Dest, const void *Src, size_t NumBytes,
                          const sycl::detail::code_location &CodeLoc =
                              sycl::detail::code_location::current());

template <typename T>
void copy(handler &CGH, const T *Src, T *Dest, size_t Count) {
  CGH.copy<T>(Src, Dest, Count);
}

template <typename T>
void copy(queue Q, const T *Src, T *Dest, size_t Count,
          const sycl::detail::code_location &CodeLoc =
              sycl::detail::code_location::current()) {
  submit(Q, [&](handler &CGH) { copy<T>(CGH, Src, Dest, Count); }, CodeLoc);
}

inline void memset(handler &CGH, void *Ptr, int Value, size_t NumBytes) {
  CGH.memset(Ptr, Value, NumBytes);
}

__SYCL_EXPORT void memset(queue Q, void *Ptr, int Value, size_t NumBytes,
                          const sycl::detail::code_location &CodeLoc =
                              sycl::detail::code_location::current());

template <typename T>
void fill(sycl::handler &CGH, T *Ptr, const T &Pattern, size_t Count) {
  CGH.fill(Ptr, Pattern, Count);
}

template <typename T>
void fill(sycl::queue Q, T *Ptr, const T &Pattern, size_t Count,
          const sycl::detail::code_location &CodeLoc =
              sycl::detail::code_location::current()) {
  submit(Q, [&](handler &CGH) { fill<T>(CGH, Ptr, Pattern, Count); }, CodeLoc);
}

inline void prefetch(handler &CGH, void *Ptr, size_t NumBytes) {
  CGH.prefetch(Ptr, NumBytes);
}

inline void prefetch(queue Q, void *Ptr, size_t NumBytes,
                     const sycl::detail::code_location &CodeLoc =
                         sycl::detail::code_location::current()) {
  submit(Q, [&](handler &CGH) { prefetch(CGH, Ptr, NumBytes); }, CodeLoc);
}

inline void mem_advise(handler &CGH, void *Ptr, size_t NumBytes, int Advice) {
  CGH.mem_advise(Ptr, NumBytes, Advice);
}

__SYCL_EXPORT void mem_advise(queue Q, void *Ptr, size_t NumBytes, int Advice,
                              const sycl::detail::code_location &CodeLoc =
                                  sycl::detail::code_location::current());

inline void barrier(handler &CGH) { CGH.ext_oneapi_barrier(); }

inline void barrier(queue Q, const sycl::detail::code_location &CodeLoc =
                                 sycl::detail::code_location::current()) {
  submit(Q, [&](handler &CGH) { barrier(CGH); }, CodeLoc);
}

inline void partial_barrier(handler &CGH, const std::vector<event> &Events) {
  CGH.ext_oneapi_barrier(Events);
}

inline void partial_barrier(queue Q, const std::vector<event> &Events,
                            const sycl::detail::code_location &CodeLoc =
                                sycl::detail::code_location::current()) {
  submit(Q, [&](handler &CGH) { partial_barrier(CGH, Events); }, CodeLoc);
}

// inline void execute_graph(queue Q, command_graph<graph_state::executable> &G,
//                           const sycl::detail::code_location &CodeLoc =
//                               sycl::detail::code_location::current()) {
//   Q.ext_oneapi_graph(G, CodeLoc);
// }

// inline void execute_graph(handler &CGH,
//                           command_graph<graph_state::executable> &G) {
//   CGH.ext_oneapi_graph(G);
// }

} // namespace khr
} // namespace _V1
} // namespace sycl
