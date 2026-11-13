// Tests recording a fork-join graph where each forked branch is recorded from
// its own thread. The primary queue records a fork point, every thread then
// concurrently records a dependent chain of kernels on its own queue, which
// transitions that queue to recording. The primary queue joins all branches.
// On out-of-order queues each chain is ordered with explicit events.

#include "../graph_threading_common.hpp"

constexpr int KernelsPerThread = 4;

void runTest(const ThreadTestConfig &Config) {
  queue Primary{Config.QueueProps};

  const unsigned NumForks = Config.NumThreads;
  std::vector<queue> ForkQueues;
  for (unsigned I = 0; I < NumForks; ++I)
    ForkQueues.emplace_back(Primary.get_context(), Primary.get_device(),
                            Config.QueueProps);

  int *Data = malloc_device<int>(Size * NumForks, Primary);
  int *Sum = malloc_device<int>(Size, Primary);
  std::vector<int> HostSum(Size);

  repeatRaceAttempts(Config, [&](const RaceAttempt &Attempt) {
    auto Graph = makeTestGraph(Primary);
    Graph.begin_recording(Primary);

    event Fork = Primary.parallel_for(range<1>{Size * NumForks}, [=](id<1> Id) {
      Data[Id] = static_cast<int>(Id[0]);
    });

    std::vector<event> Joins(NumForks);
    runConcurrently(NumForks, [&](unsigned ThreadIndex) {
      queue &ForkQueue = ForkQueues[ThreadIndex];
      const int Branch = static_cast<int>(ThreadIndex);
      int *Slice = Data + ThreadIndex * Size;
      std::vector<event> Deps{Fork};
      event Last;
      for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel) {
        Last = ForkQueue.parallel_for(range<1>{Size}, Deps, [=](id<1> Id) {
          Slice[Id] = Slice[Id] * 2 + Branch + Kernel;
        });
        Deps = orderAfter(ForkQueue, Last);
      }
      Joins[ThreadIndex] = Last;
    });

    Primary.parallel_for(range<1>{Size}, Joins, [=](id<1> Id) {
      int Total = 0;
      for (unsigned Branch = 0; Branch < NumForks; ++Branch)
        Total += Data[Branch * Size + Id[0]];
      Sum[Id] = Total;
    });

    Graph.end_recording();

    auto ExecGraph = Graph.finalize();
    for (unsigned I = 0; I < Iterations; ++I)
      Primary.ext_oneapi_graph(ExecGraph);
    Primary.wait_and_throw();
    Primary.copy(Sum, HostSum.data(), Size).wait_and_throw();

    const std::string Name = Attempt.label("Sum");
    for (size_t Index = 0; Index < Size; ++Index) {
      int Reference = 0;
      for (unsigned Branch = 0; Branch < NumForks; ++Branch) {
        int Value = static_cast<int>(Branch * Size + Index);
        for (int Kernel = 0; Kernel < KernelsPerThread; ++Kernel)
          Value = Value * 2 + static_cast<int>(Branch) + Kernel;
        Reference += Value;
      }
      assert(check_value(Index, Reference, HostSum[Index], Name));
    }
  });

  free(Data, Primary);
  free(Sum, Primary);
}

int main(int argc, char **argv) {
  runTest(ThreadTestConfig{queuePropsFromArgs(argc, argv)});
  return 0;
}
