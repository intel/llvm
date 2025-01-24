#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {
namespace khr {

template <typename CommandGroupFunc>
void submit(queue q, CommandGroupFunc &&cgf,
            const sycl::detail::code_location &code_loc =
                sycl::detail::code_location::current()) {
  sycl::ext::oneapi::experimental::submit(
      q, std::forward<CommandGroupFunc>(cgf), code_loc );
}

template <typename CommandGroupFunc>
event submit_tracked(queue q, CommandGroupFunc &&cgf,
                     const sycl::detail::code_location &code_loc =
                         sycl::detail::code_location::current()) {
  return sycl::ext::oneapi::experimental::submit_with_event(
      q, std::forward<CommandGroupFunc>(cgf), code_loc);
}

template <typename KernelType>
void launch(handler &h, range<1> r, const KernelType &k) {
  h.parallel_for<KernelType>(r, k);
}
template <typename KernelType>
void launch(handler &h, range<2> r, const KernelType &k) {
  h.parallel_for<KernelType>(r, k);
}
template <typename KernelType>
void launch(handler &h, range<3> r, const KernelType &k) {
  h.parallel_for<KernelType>(r, k);
}

template <typename KernelType>
void launch(queue q, range<1> r, const KernelType &k) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); });
}
template <typename KernelType>
void launch(queue q, range<2> r, const KernelType &k) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); });
}
template <typename KernelType>
void launch(queue q, range<3> r, const KernelType &k) {
  submit(q, [&](handler &h) { launch<KernelType>(h, r, k); });
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
void launch(queue q, range<1> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) {
    parallel_for(h, r, k, std::forward<ArgsT>(args)...);
  });
}

template <typename... ArgsT>
void launch(queue q, range<2> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) {
    parallel_for(h, r, k, std::forward<ArgsT>(args)...);
  });
}

template <typename... ArgsT>
void launch(queue q, range<3> r, const kernel &k, ArgsT &&...args) {
  submit(q, [&](handler &h) {
    parallel_for(h, r, k, std::forward<ArgsT>(args)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<1> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for<KernelType>(r, std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<2> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for<KernelType>(r, std::forward<Reductions>(reductions)..., k);
}
template <typename KernelType, typename... Reductions>
void launch_reduce(handler &h, range<3> r, const KernelType &k,
                   Reductions &&...reductions) {
  h.parallel_for<KernelType>(r, std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_reduce(queue q, range<1> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(queue q, range<2> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_reduce(queue q, range<3> r, const KernelType &k,
                   Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_reduce<KernelType>(h, r, k, std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType>
void launch_grouped(handler &h, range<1> r, range<1> size,
                    const KernelType &k) {
  h.parallel_for<KernelType>(nd_range<1>(r, size), k);
}

template <typename KernelType>
void launch_grouped(handler &h, range<2> r, range<2> size,
                    const KernelType &k) {
  h.parallel_for<KernelType>(nd_range<2>(r, size), k);
}

template <typename KernelType>
void launch_grouped(handler &h, range<3> r, range<3> size,
                    const KernelType &k) {
  h.parallel_for<KernelType>(nd_range<3>(r, size), k);
}

template <typename KernelType>
void launch_grouped(queue q, range<1> r, range<1> size, const KernelType &k) {
  submit(q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); });
}
template <typename KernelType>
void launch_grouped(queue q, range<2> r, range<2> size, const KernelType &k) {
  submit(q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); });
}
template <typename KernelType>
void launch_grouped(queue q, range<3> r, range<3> size, const KernelType &k) {
  submit(q, [&](handler &h) { launch_grouped<KernelType>(h, r, size, k); });
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
void launch_grouped(sycl::queue q, sycl::range<1> r, sycl::range<1> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename... Args>
void launch_grouped(sycl::queue q, sycl::range<2> r, sycl::range<2> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename... Args>
void launch_grouped(sycl::queue q, sycl::range<3> r, sycl::range<3> size,
                    const sycl::kernel &k, Args &&...args) {
  submit(q, [&](handler &h) {
    launch_grouped(h, r, size, k, std::forward<Args>(args)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<1> r,
                           sycl::range<1> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for<KernelType>(nd_range<1>(r, size),
                             std::forward<Reductions>(reductions)..., k);
}
template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<2> r,
                           sycl::range<2> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for<KernelType>(nd_range<2>(r, size),
                             std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::handler &h, sycl::range<3> r,
                           sycl::range<3> size, const KernelType &k,
                           Reductions &&...reductions) {
  h.parallel_for<KernelType>(nd_range<3>(r, size),
                             std::forward<Reductions>(reductions)..., k);
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::queue q, sycl::range<1> r, sycl::range<1> size,
                           const KernelType &k, Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::queue q, sycl::range<2> r, sycl::range<2> size,
                           const KernelType &k, Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType, typename... Reductions>
void launch_grouped_reduce(sycl::queue q, sycl::range<3> r, sycl::range<3> size,
                           const KernelType &k, Reductions &&...reductions) {
  submit(q, [&](handler &h) {
    launch_grouped_reduce<KernelType>(h, r, size, k,
                                      std::forward<Reductions>(reductions)...);
  });
}

template <typename KernelType>
void single_task(handler &h, const KernelType &k) {
  h.single_task<KernelType>(k);
}

template <typename KernelType>
void launch_task(sycl::queue q, const KernelType &k,
                 const sycl::detail::code_location &code_loc =
                     sycl::detail::code_location::current()) {
  submit(q, [&](handler &h) { single_task<KernelType>(h, k); }, code_loc);
}

template <typename... Args>
void launch_task(sycl::handler &h, const sycl::kernel &k, Args &&...args) {
  h.set_args<Args...>(std::forward<Args>(args)...);
  h.single_task(k);
}

template <typename... Args>
void launch_task(queue q, const kernel &k, Args &&...args) {
  submit(q,
         [&](handler &h) { launch_task(h, k, std::forward<Args>(args)...); });
}
} // namespace khr
} // namespace _V1
} // namespace sycl
