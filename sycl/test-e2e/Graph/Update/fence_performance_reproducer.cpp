// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

const size_t N = 10000;
const size_t NumNodes = 1000;
void initializeGraphA(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    std::vector<exp_ext::node> &Nodes,
    std::vector<exp_ext::dynamic_parameter<int *>> &DynamicParams, int *PtrA) {

  for (int i = 0; i < NumNodes; ++i) {
    DynamicParams.push_back(exp_ext::dynamic_parameter(Graph, PtrA));
  }

  auto CurrentNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DynamicParams[0]);
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        PtrA[i] = i;
      }
    });
  });
  Nodes.push_back(CurrentNode);

  for (int i = 1; i < NumNodes; ++i) {
    auto Node = Graph.add(
        [&](handler &cgh) {
          cgh.set_arg(0, DynamicParams[i]);
          cgh.single_task([=]() {
            for (size_t i = 0; i < N; i++) {
              PtrA[i] = i;
            }
          });
        },
        exp_ext::property::node::depends_on(CurrentNode));
    CurrentNode = Node;
    Nodes.push_back(CurrentNode);
  }
}

void initializeGraphB(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    std::vector<exp_ext::node> &Nodes,
    std::vector<exp_ext::dynamic_parameter<int *>> &DynamicParams, int *PtrA) {

  for (int i = 0; i < NumNodes; ++i) {
    DynamicParams.push_back(exp_ext::dynamic_parameter(Graph, PtrA));
  }

  int C = 3232;
  auto CurrentNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DynamicParams[0]);
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        PtrA[i] = i + C;
      }
    });
  });
  Nodes.push_back(CurrentNode);

  for (int i = 1; i < NumNodes; ++i) {
    auto Node = Graph.add(
        [&](handler &cgh) {
          cgh.set_arg(0, DynamicParams[i]);
          cgh.single_task([=]() {
            for (size_t i = 0; i < N; i++) {
              PtrA[i] = i + C;
            }
          });
        },
        exp_ext::property::node::depends_on(CurrentNode));
    CurrentNode = Node;
    Nodes.push_back(CurrentNode);
  }
}

int main() {
  queue Queue{};

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  std::vector<exp_ext::dynamic_parameter<int *>> DynamicParamsA{};
  std::vector<exp_ext::dynamic_parameter<int *>> DynamicParamsB{};
  std::vector<exp_ext::node> NodesA{};
  std::vector<exp_ext::node> NodesB{};

  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);
  int *PtrA2 = malloc_device<int>(N, Queue);
  int *PtrB2 = malloc_device<int>(N, Queue);

  //  for (int i = 0; i < 5; ++i) {
  //    Queue.submit([&](handler &cgh) {
  //      cgh.single_task([=]() {
  //        for (size_t i = 0; i < N; i++) {
  //          PtrA[i] = i;
  //        }
  //      });
  //      std::cout << "SubmittedEager" << std::endl;
  ////      int *PtrCC2 = malloc_shared<int>(1, Queue);
  ////      std::cout << PtrCC2[0] << std::endl;
  ////      sycl::free(PtrCC2, Queue);
  //    });
  //
  //    Queue2.submit([&](handler &cgh) {
  //      cgh.single_task([=]() {
  //        for (size_t i = 0; i < N; i++) {
  //          PtrB[i] = i;
  //        }
  //      });
  //    });
  //    std::cout << "SubmittedEager" << std::endl;
  //  }

  initializeGraphA(GraphA, NodesA, DynamicParamsA, PtrA);
  initializeGraphB(GraphB, NodesB, DynamicParamsB, PtrB);

  auto ExecGraphA = GraphA.finalize(exp_ext::property::graph::updatable{});
  auto ExecGraphB = GraphB.finalize(exp_ext::property::graph::updatable{});

  auto NextPtrA = PtrA2;
  auto NextPtrB = PtrB2;
  //  for (int i = 0; i < 2; ++i) {
  //    for (int j = 0; j < NumNodes; ++j) {
  //      DynamicParamsA[j].update(NextPtrA);
  //      ExecGraphA.update(NodesA[j]);
  //    }
  //    for (int j = 0; j < NumNodes; ++j) {
  //      DynamicParamsB[j].update(NextPtrB);
  //      ExecGraphB.update(NodesB[j]);
  //    }
  //
  //    Queue.ext_oneapi_graph(ExecGraphA);
  //
  //    for (int j = 0; j < NumNodes; ++j) {
  //      DynamicParamsB[j].update(NextPtrB);
  //      ExecGraphB.update(NodesB[j]);
  //    }
  //    Queue.ext_oneapi_graph(ExecGraphB).wait();
  //
  //    NextPtrA = NextPtrA == PtrA ? PtrA2 : PtrA;
  //    NextPtrB = NextPtrB == PtrB ? PtrB2 : PtrB;
  //  }

  Queue.ext_oneapi_graph(ExecGraphA);
  Queue.ext_oneapi_graph(ExecGraphB).wait();

  NextPtrA = NextPtrA == PtrA ? PtrA2 : PtrA;
  NextPtrB = NextPtrB == PtrB ? PtrB2 : PtrB;

  for (int j = 0; j < NumNodes; ++j) {
    DynamicParamsA[j].update(NextPtrA);
    ExecGraphA.update(NodesA[j]);
  }
//  for (int j = 0; j < NumNodes; ++j) {
//    DynamicParamsB[j].update(NextPtrB);
//    ExecGraphB.update(NodesB[j]);
//  }

  Queue.ext_oneapi_graph(ExecGraphA);
//  Queue.ext_oneapi_graph(ExecGraphB);

  for (int j = 0; j < NumNodes; ++j) {
    DynamicParamsB[j].update(NextPtrB);
    ExecGraphB.update(NodesB[j]);
  }
  Queue.ext_oneapi_graph(ExecGraphB).wait();



  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrA2, Queue);
  sycl::free(PtrB2, Queue);
  return 0;
}
