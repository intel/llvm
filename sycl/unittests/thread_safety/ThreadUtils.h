#include <thread>
#include <vector>

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

  ~ThreadPool() { wait(); }

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
