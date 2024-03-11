#include "FindPrimesSYCL.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#define DEBUG 0

using namespace std;

class gpu_tsk {
public:
  gpu_tsk(work &w) : m_work(w) {}
  void operator()() const {
    find_prime_s(&m_work);
    m_work.run = (m_work.end_time - m_work.start_time) / 1000000000.0f;
    m_work.wait = (m_work.start_time - m_work.submit_time) / 1000000000.0f;
    m_work.elapsed = m_work.run + m_work.wait;
  }

private:
  work &m_work;
};

static std::atomic<float> cpusum{0.};

bool sortBySubmitTime(work &w1, work &w2) {
  return w1.submit_time < w2.submit_time;
}

int main(int argc, char *argv[]) {

  int opt;
  bool lock{true};
  bool sharedQueue{true};
  // int nthreadsGPU = 1;
  int nthreadsGPU = 8;
  int arr_size = 20;
  int iter_gpu = 200;
  unsigned int gpu_dev = 999;
  unsigned int nitems = 0;
  bool passed = true;

  // Size of arrays
  size_t N = 1 << arr_size;

  sycl::device sel_dev;
  std::mutex *queueLock{nullptr};

  if (lock) {
    queueLock = new std::mutex;
  }

  if (nitems == 0) {
    nitems = N;
  }
  if (nitems > N) {
    nitems = N;
  }

#ifdef DEBUG
  cout << "NThreads: GPU " << nthreadsGPU << endl;
  cout << "   shared Queue: " << std::boolalpha << sharedQueue << endl;
  cout << "   array size: " << N << " (1<<" << arr_size << ")" << endl;
  cout << "   SyCL items: " << nitems << endl;
  cout << "   iter GPU:   " << iter_gpu << "\n";
#endif

  std::vector<sycl::device> dlist;
  if (gpu_dev == 999) {
    try {
      auto sel = sycl::gpu_selector();
      sel_dev = sel.select_device();
    } catch (...) {
      cout << "no gpu device found\n";
    }
  } else {
    if (gpu_dev > dlist.size() - 1) {
      cout << "ERROR: selected device index [" << gpu_dev << "] is too large\n";
      exit(1);
    }
    sel_dev = dlist[gpu_dev];
  }
  std::cout << "selected dev: " << sel_dev.get_info<sycl::info::device::name>()
            << "\n";

  std::cout << std::endl;

  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  auto property_list =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue deviceQueue(sel_dev, exception_handler, property_list);

  std::cout << "\n";

  vector<work> vwork;

  std::chrono::duration<double> diff{0};

  work w[nthreadsGPU];
  for (int i = 0; i < nthreadsGPU; ++i) {
    w[i].id = i;
    w[i].size = N;
    w[i].niter = iter_gpu;
    w[i].nitems = nitems;
    w[i].VRI.resize(N);
    w[i].success = false;
    w[i].queueLock = queueLock;
    if (sharedQueue) {
      w[i].deviceQueue = &deviceQueue;
    } else {
      w[i].deviceQueue = new sycl::queue(deviceQueue);
    }
    vwork.push_back(w[i]);
  }

  std::vector<thread> tv, tc;
  tv.reserve(nthreadsGPU);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < nthreadsGPU; ++i) {
    tv.push_back(thread{gpu_tsk(vwork[i])});
  }

  for (auto &v : tv) {
    v.join();
  }

  auto stop = std::chrono::high_resolution_clock::now();

  float avgtime{0.};

  std::sort(vwork.begin(), vwork.end(), sortBySubmitTime);

  auto mint = vwork[0].submit_time;
  auto minstart = vwork[0].start;

#ifdef DEBUG
  printf("%26s %4s   %4s   %4s   %4s\n", " ", "subm", "strt", "end", "run");
#endif

  for (auto &w : vwork) {
    if (w.submit_time < mint) {
      mint = w.submit_time;
    }
    if (w.start < minstart) {
      minstart = w.start;
    }

    int nPrimes = 0;
    for (size_t i = 2; i < w.size; ++i) {
      if (w.VRI[i]) {
        nPrimes++;
      }
    }
    avgtime += w.run;

#ifdef DEBUG
    std::chrono::duration<double> d1 = w.start - minstart;
    std::chrono::duration<double> d2 = w.stop - minstart;
    std::chrono::duration<double> d3 = w.stop - w.start;

    std::cout << "GPU[" << w.id << "]:  nPrimes: " << nPrimes << "  "
              << std::fixed << std::setprecision(2) << std::setw(6)
              << float(w.submit_time - mint) / 1000000000. << " "
              << std::setw(6) << float(w.start_time - mint) / 1000000000. << " "
              << std::setw(6) << float(w.end_time - mint) / 1000000000. << " "
              << std::setw(6) << float(w.end_time - w.start_time) / 1000000000.
              << "    " << std::setw(6) << d1.count() << " " << std::setw(6)
              << d2.count() << " " << std::setw(6) << d3.count() << std::endl;
#endif

    if (!sharedQueue)
      delete w.deviceQueue;

    if (nPrimes != 82024) {
      passed = false;
    }
  }

#ifdef DEBUG
  diff = (stop - start);
  std::cout << "GPU average time: " << avgtime / vwork.size() << std::endl;
  std::cout << "total time: " << diff.count() << std::endl;
#endif

  delete queueLock;

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
    exit(1);
  }
  return 0;
}
