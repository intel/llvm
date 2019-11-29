#include "ThreadUtils.h"
#include <iostream>

static void *threadMainBody(Thread *thread, size_t id) {
  thread->body(id);
  return NULL;
}

void Thread::start(size_t id) {
  MThread = std::move(std::thread(threadMainBody, this, id));
}

void Thread::wait() { MThread.join(); }

void Thread::body(size_t id) { MTask->taskBody(id); }

ThreadPool::ThreadPool(ParallelTask *p) : MTask(p) {}

void ThreadPool::initialize(int size) {
  for (int i = 0; i < size; ++i) {
    MThreadPool.push_back(new Thread(MTask));
  }
}

void ThreadPool::start() {
  for (std::size_t i = 0; i < MThreadPool.size(); ++i) {
    MThreadPool[i]->start(i);
  }
}

void ThreadPool::wait() {
  for (auto it : MThreadPool) {
    it->wait();
  }
}

void ParallelTask::execute(int threadCount) {
  try {
    MPool.initialize(threadCount);
    MPool.start();
    MPool.wait();
  } catch (std::exception &ex) {
    std::cerr << ex.what();
  } catch (...) {
    std::cerr << "Unknown exception";
  }
}
