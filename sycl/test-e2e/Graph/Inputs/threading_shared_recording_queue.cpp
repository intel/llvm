// Tests multiple threads concurrently recording kernel chains to the same
// recording queue. The order in which threads' submissions interleave is
// undefined, so each thread only touches its own slice of the data. Each
// thread's chain of kernels is not commutative, so the chain must be ordered
// for the result to be correct. On out-of-order queues the chain is ordered
// with explicit events.

#include "../graph_threading_common.hpp"

constexpr int KernelsPerThread = 4;

void runTest(const ThreadTestConfig &Config) {
  queue Queue{Config.QueueProps};

  const unsigned NumThreads = Config.NumThreads;
  int *Data = malloc_device<int>(Size * NumThreads, Queue);
  std::vector<int> HostData(Size * NumThreads);

  repeatRaceAttempts(Config, [&](const RaceAttempt &Attempt) {
    auto Graph = makeTestGraph(Queue);
    Graph.begin_recording(Queue);

    event InitEvent =
        Queue.parallel_for(range<1>{Size * NumThreads}, [=](id<1> Id) {
          Data[Id] = static_cast<int>(Id[0]);
        });

    runConcurrently(NumThreads, [&](unsigned ThreadIndex) {
      const int Thread = static_cast<int>(ThreadIndex);
      int *Slice = Data + ThreadIndex * Size;
      event Last = InitEvent;
      for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
        Last = Queue.parallel_for(
            range<1>{Size}, orderAfter(Queue, Last),
            [=](id<1> Id) { Slice[Id] = Slice[Id] * 2 + Thread + Kernel; });
    });

    Graph.end_recording(Queue);

    auto ExecGraph = Graph.finalize();
    for (unsigned I = 0; I < Iterations; ++I)
      Queue.ext_oneapi_graph(ExecGraph);
    Queue.wait_and_throw();
    Queue.copy(Data, HostData.data(), HostData.size()).wait_and_throw();

    const std::string Name = Attempt.label("Data");
    for (size_t Index = 0; Index < HostData.size(); ++Index) {
      const int Thread = static_cast<int>(Index / Size);
      int Reference = static_cast<int>(Index);
      for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
        Reference = Reference * 2 + Thread + Kernel;
      assert(check_value(Index, Reference, HostData[Index], Name));
    }
  });

  free(Data, Queue);
}

int main(int argc, char **argv) {
  runTest(ThreadTestConfig{queuePropsFromArgs(argc, argv)});
  return 0;
}
