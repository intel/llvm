// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES-INTEL-DRIVER: lin: 37561, win: 101.8724

// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Regression test for a lock-order inversion in native recording between the
// queue mutex and the graph mutex.
//
// Two lock orders exist on the native-recording paths:
//   * graph -> queue: end_recording() takes graph_impl::MMutex then the queue
//     mutex (via queue_impl::endNativeRecording()).
//   * queue -> graph: submitting a restricted host task runs
//   handler::finalize()
//     under the queue mutex, which used to take graph_impl::MMutex via
//     graph_impl::addNativeHostTaskCallback().
//
// Running both concurrently on the same queue and graph could deadlock.
// The fix gives the native host-task callback list its own leaf mutex so the
// submit path never acquires graph_impl::MMutex.

#include "../../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

#include <thread>

constexpr size_t HostTasksPerIter = 32;

int main() {
  queue Queue{property::queue::in_order{}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  for (size_t Iter = 0; Iter < Iterations; ++Iter) {
    exp_ext::command_graph Graph{
        Context, Device, {exp_ext::property::graph::enable_native_recording{}}};

    Graph.begin_recording(Queue);

    // Tighten the race window: the submitting thread and the ending thread
    // start their contended operation at (nearly) the same time.
    Barrier Sync{2};

    // Submitting thread takes the queue -> graph lock order for every
    // restricted host task captured into the native graph.
    std::thread Submitter([&]() {
      Sync.wait();
      for (size_t i = 0; i < HostTasksPerIter; ++i) {
        // Side-effect-free host task: valid whether it is captured (while
        // recording) or run as a normal host task (after recording ends).
        exp_ext::host_task(Queue, [] {});
      }
    });

    // Main thread takes the graph -> queue lock order.
    Sync.wait();
    Graph.end_recording(Queue);

    Submitter.join();
    Queue.wait_and_throw();
  }

  return 0;
}
