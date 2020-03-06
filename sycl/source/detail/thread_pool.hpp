#pragma once

#include <list>
#include <thread>

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class ThreadPool {
  std::list<std::thread> MLaunchedThreads;
public:
  ThreadPool() {}
  ~ThreadPool() {
    for (std::thread &Thr : MLaunchedThreads) {
      if (Thr.joinable())
        Thr.join();
    }
  }

  template <typename FuncT, typename... ArgsT>
  void submit(FuncT &&Func, ArgsT... Args) {
    MLaunchedThreads.emplace_back(Func, Args...);
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
