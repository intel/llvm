#include <detail/scheduler/scheduler.hpp>

#include <condition_variable> // std::conditional_variable
#include <mutex>              // std::mutex, std::unique_lock
#include <thread>             // std::thread
#include <utility>            // std::forward
#include <vector>             // std::vector

/* Single use thread barrier which makes threads wait until defined number of
 * threads reach it.
 * std:barrier should be used instead once compiler is moved to C++20 standard.
 */
class Barrier {
public:
  Barrier() = delete;
  explicit Barrier(std::size_t count) : threadNum(count) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    if (--threadNum == 0) {
      cv.notify_all();
    } else {
      cv.wait(lock, [this] { return threadNum == 0; });
    }
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
  std::size_t threadNum;
};

class ThreadPool {
public:
  ThreadPool() = delete;
  ThreadPool(ThreadPool &) = delete;

  template <typename Func>
  ThreadPool(std::size_t N, Func func) {
    for (std::size_t i = 0; i < N; ++i) {
      enqueue(func, i);
    }
  }

  template <typename Func, typename... Funcs>
  ThreadPool(Func &&func, Funcs &&... funcs) {
    constexpr int N = sizeof...(funcs);
    enqueue(std::forward<Func>(func), N);
    enqueueHelper<N>(std::forward<Funcs>(funcs)...);
  }

  ~ThreadPool() {
    try {
      wait();
    } catch (std::exception &e) {
      std::cerr << "exception in ~ThreadPool" << e.what() << std::endl;
    }
  }

private:
  template <int N, typename Func, typename... Funcs>
  void enqueueHelper(Func &&func, Funcs &&... funcs) {
    enqueue(std::forward<Func>(func), N - 1);
    enqueueHelper<N - 1>(std::forward<Funcs>(funcs)...);
  }

  template <int N>
  void enqueueHelper() {}

  template <typename Func, typename... Args>
  void enqueue(Func &&func, Args &&... args) {
    MThreadPool.emplace_back(std::forward<Func>(func),
                             std::forward<Args>(args)...);
  }

  void wait() {
    for (auto &t : MThreadPool) {
      t.join();
    }
  }

  std::vector<std::thread> MThreadPool;
};
