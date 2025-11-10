// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for enable_native_recording property using queue recording mode

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  // Create a queue with immediate command list property for native recording
  queue Queue{{
      property::queue::in_order{},
      ext::intel::property::queue::immediate_command_list{}
  }};

  // Create a graph with native recording enabled for improved performance
  auto MyProperties = property_list{
      exp_ext::property::graph::enable_native_recording{}
  };
  
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device(), MyProperties};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  // Use queue recording mode to create the graph
  Graph.begin_recording(Queue);

  // Record initialization kernel
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] = static_cast<int>(idx);
    });
  });

  // Record computation kernel
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] = Data[idx] * 2;
    });
  });

  Graph.end_recording(Queue);

  // Finalize and execute the graph
  auto ExecutableGraph = Graph.finalize();
  
  Queue.submit([&](handler &CGH) { 
    CGH.ext_oneapi_graph(ExecutableGraph); 
  });
  
  Queue.wait();

  // Verify results
  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i) * 2;
    assert(check_value(i, Expected, HostData[i], "HostData"));
  }

  free(Data, Queue);

  return 0;
}