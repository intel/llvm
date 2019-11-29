#include <thread>
#include <vector>

class ParallelTask;

class Thread {
public:
  Thread(ParallelTask *Ptr) : MTask(Ptr) {}
  void start(size_t id);
  void wait();
  void body(size_t id);

private:
  std::thread MThread;
  ParallelTask *MTask;
};

class ThreadPool {
public:
  ThreadPool(ParallelTask *p);
  void initialize(int size);
  void start();
  void wait();

private:
  std::vector<Thread *> MThreadPool;
  ParallelTask *MTask;
};

class ParallelTask {
  friend class ThreadPool;

public:
  ParallelTask() : MPool(this) {}

  void execute(int threadCount);

  virtual void taskBody(std::size_t id) = 0;

private:
  ThreadPool MPool;
};
