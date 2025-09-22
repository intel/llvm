// REQUIRES: aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl
// UNSUPPORTED-INTENDED: OpenCL currently has limited support for command
// buffers
//
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests prefetch functionality in enqueue functions

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

static constexpr int N = 100;
static constexpr int Pattern = 42;

int main() {
  queue Q{};

  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);
  for (int i = 0; i < N; i++)
    Src[i] = Pattern;

  {
    exp_ext::command_graph Graph{Q.get_context(), Q.get_device(), {}};

    Graph.begin_recording(Q);

    // Test submitting host-to-device prefetch
    event TestH2D = exp_ext::submit_with_event(
        Q, [&](handler &CGH) { exp_ext::prefetch(CGH, Src, sizeof(int) * N); });

    exp_ext::submit(Q, [&](handler &CGH) {
      CGH.depends_on(TestH2D);
      exp_ext::parallel_for(CGH, range<1>(N),
                            [=](id<1> i) { Dst[i] = Src[i] * 2; });
    });

    Graph.end_recording();

    auto GraphExec = Graph.finalize();

    exp_ext::execute_graph(Q, GraphExec);
    Q.wait_and_throw();
  }

  // Check host-to-device prefetch results
  for (int i = 0; i < N; i++)
    assert(check_value(i, Pattern * 2, Dst[i], "Dst"));

  {
    exp_ext::command_graph Graph{Q.get_context(), Q.get_device(), {}};

    Graph.begin_recording(Q);

    // Test submitting device-to-host prefetch
    event TestD2H = exp_ext::submit_with_event(Q, [&](handler &CGH) {
      exp_ext::parallel_for(CGH, range<1>(N),
                            [=](id<1> i) { Dst[i] = Src[i] + 1; });
    });

    exp_ext::submit(Q, [&](handler &CGH) {
      CGH.depends_on(TestD2H);
      exp_ext::prefetch(CGH, Dst, sizeof(int) * N,
                        exp_ext::prefetch_type::host);
    });

    Graph.end_recording();

    auto GraphExec = Graph.finalize();

    exp_ext::execute_graph(Q, GraphExec);
    Q.wait_and_throw();
  }

  // Check device-to-host prefetch results
  for (int i = 0; i < N; i++)
    assert(check_value(i, Pattern + 1, Dst[i], "Dst"));

  free(Src, Q);
  free(Dst, Q);

  return 0;
}
