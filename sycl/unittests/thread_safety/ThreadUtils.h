#include <thread>
#include <vector>

class ThreadPool {
public:
  void clear() { MThreadPool.clear(); }

  template <typename Func, typename... Args>
  void enqueueNTimes(std::size_t N, Func &&func, Args &&... args) {
    for (std::size_t i = 0; i < N; ++i)
      enqueue(std::forward<Func>(func), std::forward<Args>(args)...);
  }

  template <typename Func, typename... Args>
  void enqueue(Func &&func, Args &&... args) {
    MThreadPool.push_back(
        std::thread(std::forward<Func>(func), std::forward<Args>(args)...));
  }

  void wait() {
    for (auto &t : MThreadPool) {
      t.join();
    }
  }

private:
  std::vector<std::thread> MThreadPool;
};
