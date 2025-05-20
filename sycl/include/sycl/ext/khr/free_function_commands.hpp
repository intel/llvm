#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
namespace khr {

template <typename CommandGroupFunc>
void submit(const queue &q, CommandGroupFunc &&cgf,
            const sycl::detail::code_location &codeLoc =
                sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::submit(
      q, std::forward<CommandGroupFunc>(cgf), codeLoc);
}

template <typename CommandGroupFunc>
event submit_tracked(queue q, CommandGroupFunc &&cgf,
                     const sycl::detail::code_location &codeLoc =
                         sycl::detail::code_location::current()) {
  return sycl::ext::oneapi::experimental::submit_with_event(
      q, std::forward<CommandGroupFunc>(cgf), codeLoc);
}

template <typename KernelType>
void launch(handler &h, range<1> r, const KernelType &k) {
  h.parallel_for(r, k);
}
template <typename KernelType>
void launch(handler &h, range<2> r, const KernelType &k) {
  h.parallel_for(r, k);
}
template <typename KernelType>
void launch(handler &h, range<3> r, const KernelType &k) {
  h.parallel_for(r, k);
}

template <typename KernelType>
void launch(const queue &q, range<1> r, const KernelType &k,
            const sycl::detail::code_location &codeLoc =
                sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); }, codeLoc);
}
template <typename KernelType>
void launch(const queue &q, range<2> r, const KernelType &k,
            const sycl::detail::code_location &codeLoc =
                sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); }, codeLoc);
}
template <typename KernelType>
void launch(const queue &q, range<3> r, const KernelType &k,
            const sycl::detail::code_location &codeLoc =
                sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); }, codeLoc);
}

template <typename... ArgsT>
void launch(handler &h, range<1> r, const kernel &k, ArgsT &&...args) {
  h.set_args<ArgsT...>(std::forward<ArgsT>(args)...);
  h.parallel_for(r, k);
}

template <typename... ArgsT>
void launch(handler &h, range<2> r, const kernel &k, ArgsT &&...args) {
  h.set_args<ArgsT...>(std::forward<ArgsT>(args)...);
  h.parallel_for(r, k);
}

template <typename... ArgsT>
void launch(handler &h, range<3> r, const kernel &k, ArgsT &&...args) {
  h.set_args<ArgsT...>(std::forward<ArgsT>(args)...);
  h.parallel_for(r, k);
}

template <typename... ArgsT>
void launch(const queue &q, range<1> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) { launch(h, r, k, std::forward<ArgsT>(args)...); });
}

template <typename... ArgsT>
void launch(const queue &q, range<2> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) { launch(h, r, k, std::forward<ArgsT>(args)...); });
}

template <typename... ArgsT>
void launch(const queue &q, range<3> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) { launch(h, r, k, std::forward<ArgsT>(args)...); });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<1> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for(r, std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<2> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for(r, std::forward<Reductions>(reductions)..., k);
}
template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<3> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for(r, std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_reduce(const queue &q, range<1> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(const queue &q, range<2> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(const queue &q, range<3> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType>
void launch_grouped(handler &h, range<1> r, range<1> size,
                    const KernelType &k) {
  h.parallel_for(nd_range<1>(r, size), k);
}

template <typename KernelType>
void launch_grouped(handler &h, range<2> r, range<2> size,
                    const KernelType &k) {
  h.parallel_for(nd_range<2>(r, size), k);
}

template <typename KernelType>
void launch_grouped(handler &h, range<3> r, range<3> size,
                    const KernelType &k) {
  h.parallel_for(nd_range<3>(r, size), k);
}

template <typename KernelType>
void launch_grouped(const queue &q, range<1> r, range<1> size,
                    const KernelType &k,
                    const sycl::detail::code_location &codeLoc =
                        sycl::detail::code_location::current()) {
  submit(
      q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); },
      codeLoc);
}
template <typename KernelType>
void launch_grouped(const queue &q, range<2> r, range<2> size,
                    const KernelType &k,
                    const sycl::detail::code_location &codeLoc =
                        sycl::detail::code_location::current()) {
  submit(
      q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); },
      codeLoc);
}
template <typename KernelType>
void launch_grouped(const queue &q, range<3> r, range<3> size,
                    const KernelType &k,
                    const sycl::detail::code_location &codeLoc =
                        sycl::detail::code_location::current()) {
  submit(
      q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); },
      codeLoc);
}

template <typename... Args>
void launch_grouped(sycl::handler &h, sycl::range<1> r, sycl::range<1> size,
                    const sycl::kernel &k, Args &&...args) {
  h.set_args<Args...>(std::forward<Args>(args)...);
  h.parallel_for(nd_range<1>(r, size), k);
}

template <typename... Args>
void launch_grouped(sycl::handler &h, sycl::range<2> r, sycl::range<2> size,
                    const sycl::kernel &k, Args &&...args) {
  h.set_args<Args...>(std::forward<Args>(args)...);
  h.parallel_for(nd_range<2>(r, size), k);
}

template <typename... Args>
void launch_grouped(sycl::handler &h, sycl::range<3> r, sycl::range<3> size,
                    const sycl::kernel &k, Args &&...args) {
  h.set_args<Args...>(std::forward<Args>(args)...);
  h.parallel_for(nd_range<3>(r, size), k);
}

template <typename... Args>
void launch_grouped(const sycl::queue &q, sycl::range<1> r, sycl::range<1> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename... Args>
void launch_grouped(const sycl::queue &q, sycl::range<2> r, sycl::range<2> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename... Args>
void launch_grouped(const sycl::queue &q, sycl::range<3> r, sycl::range<3> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<1> r,
                           sycl::range<1> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for(nd_range<1>(r, size), std::forward<Reductions>(reductions)...,
                 k);
}
template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<2> r,
                           sycl::range<2> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for(nd_range<2>(r, size), std::forward<Reductions>(reductions)...,
                 k);
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<3> r,
                           sycl::range<3> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for(nd_range<3>(r, size), std::forward<Reductions>(reductions)...,
                 k);
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(const sycl::queue &q, sycl::range<1> r,
                           sycl::range<1> size, const KernelType &k,
                           Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(const sycl::queue &q, sycl::range<2> r,
                           sycl::range<2> size, const KernelType &k,
                           Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(const sycl::queue &q, sycl::range<3> r,
                           sycl::range<3> size, const KernelType &k,
                           Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType>
void launch_task(handler &h, const KernelType &k) {
  h.single_task(k);
}

template <typename KernelType>
void launch_task(const sycl::queue &q, const KernelType &k,
                 const sycl::detail::code_location &codeLoc =
                     sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { launch_task<KernelType>(h, k); }, codeLoc);
}

template <typename... Args>
void launch_task(sycl::handler &h, const sycl::kernel &k, Args &&...args) {
  h.set_args<Args...>(std::forward<Args>(args)...);
  h.single_task(k);
}

template <typename... Args>
void launch_task(const queue &q, const kernel &k, Args &&...args) {
  submit(q,
         [&](handler &h) { launch_task(h, k, std::forward<Args>(args)...); });
}

inline void memcpy(handler &h, void *dest, const void *src, size_t numBytes) {
  h.memcpy(dest, src, numBytes);
}
inline void memcpy(const queue &q, void *dest, const void *src, size_t numBytes,
                   const sycl::detail::code_location &codeLoc =
                       sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::memcpy(q, dest, src, numBytes, codeLoc);
}

template <typename T>
void copy(handler &h, const T *src, T *dest, size_t count) {
  h.copy(src, dest, count);
}

template <typename T>
void copy(const queue &q, const T *src, T *dest, size_t count,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { copy(h, src, dest, count); }, codeLoc);
}

template <typename SrcT, typename DestT, int DestDims, access_mode DestMode>
void copy(handler &h, const SrcT *src,
          accessor<DestT, DestDims, DestMode, target::device> dest) {
  h.copy(src, dest);
}

template <typename SrcT, typename DestT, int DestDims, access_mode DestMode>
void copy(handler &h, std::shared_ptr<SrcT> src,
          accessor<DestT, DestDims, DestMode, target::device> dest) {
  h.copy(src, dest);
}

template <typename SrcT, typename DestT, int DestDims, access_mode DestMode>
void copy(const queue &q, const SrcT *src,
          accessor<DestT, DestDims, DestMode, target::device> dest,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(dest);
        copy(h, src, dest);
      },
      codeLoc);
}

template <typename SrcT, typename DestT, int DestDims, access_mode DestMode>
void copy(const queue &q, std::shared_ptr<SrcT> src,
          accessor<DestT, DestDims, DestMode, target::device> dest,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(dest);
        copy(h, src, dest);
      },
      codeLoc);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT>
void copy(handler &h, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          DestT *dest) {
  h.copy(src, dest);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT>
void copy(handler &h, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          std::shared_ptr<DestT> dest) {
  h.copy(src, dest);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT>
void copy(const queue &q, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          DestT *dest,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(src);
        copy(h, src, dest);
      },
      codeLoc);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT>
void copy(const queue &q, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          std::shared_ptr<DestT> dest,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(src);
        copy(h, src, dest);
      },
      codeLoc);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT,
          int DestDims, access_mode DestMode>
void copy(handler &h, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          accessor<DestT, DestDims, DestMode, target::device> dest) {
  h.copy(src, dest);
}

template <typename SrcT, int SrcDims, access_mode SrcMode, typename DestT,
          int DestDims, access_mode DestMode>
void copy(const queue &q, accessor<SrcT, SrcDims, SrcMode, target::device> src,
          accessor<DestT, DestDims, DestMode, target::device> dest,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(src);
        h.require(dest);
        copy(h, src, dest);
      },
      codeLoc);
}
inline void memset(handler &h, void *ptr, int value, size_t numBytes) {
  h.memset(ptr, value, numBytes);
}

inline void memset(const queue &q, void *ptr, int value, size_t numBytes,
                   const sycl::detail::code_location &codeLoc =
                       sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::memset(q, ptr, value, numBytes, codeLoc);
}

template <typename T>
void fill(handler &h, T *ptr, const T &pattern, size_t count) {
  h.fill(ptr, pattern, count);
}

template <typename T, int Dims, access_mode Mode>
void fill(handler &h, accessor<T, Dims, Mode, target::device> dest,
          const T &src) {
  h.fill(dest, src);
}

template <typename T>
void fill(const queue &q, T *ptr, const T &pattern, size_t count,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { fill(h, ptr, pattern, count); }, codeLoc);
}

template <typename T, int Dims, access_mode Mode>
void fill(const queue &q, accessor<T, Dims, Mode, target::device> dest,
          const T &src,
          const sycl::detail::code_location &codeLoc =
              sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(dest);
        fill(h, dest, src);
      },
      codeLoc);
}

template <typename T, int Dims, access_mode Mode>
void update_host(handler &h, accessor<T, Dims, Mode, target::device> acc) {
  h.update_host(acc);
}

template <typename T, int Dims, access_mode Mode>
void update_host(const queue &q, accessor<T, Dims, Mode, target::device> acc,
                 const sycl::detail::code_location &codeLoc =
                     sycl::detail::code_location::current()) {
  submit(
      q,
      [&](handler &h) {
        h.require(acc);
        update_host(h, acc);
      },
      codeLoc);
}
inline void prefetch(handler &h, void *ptr, size_t numBytes) {
  h.prefetch(ptr, numBytes);
}

inline void prefetch(const queue &q, void *ptr, size_t numBytes,
                     const sycl::detail::code_location &codeLoc =
                         sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { prefetch(h, ptr, numBytes); }, codeLoc);
}

inline void mem_advise(handler &h, void *ptr, size_t numBytes, int advice) {
  h.mem_advise(ptr, numBytes, advice);
}

inline void mem_advise(const queue &q, void *ptr, size_t numBytes, int advice,
                       const sycl::detail::code_location &codeLoc =
                           sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::mem_advise(q, ptr, numBytes, advice,
                                              codeLoc);
}

inline void command_barrier(handler &h) { h.ext_oneapi_barrier(); }

inline void command_barrier(const queue &q,
                            const sycl::detail::code_location &codeLoc =
                                sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { command_barrier(h); }, codeLoc);
}

inline void event_barrier(handler &h, const std::vector<event> &events) {
  h.ext_oneapi_barrier(events);
}

inline void event_barrier(const queue &q, const std::vector<event> &events,
                          const sycl::detail::code_location &codeLoc =
                              sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { event_barrier(h, events); }, codeLoc);
}

} // namespace khr
#endif
} // namespace _V1
} // namespace sycl
