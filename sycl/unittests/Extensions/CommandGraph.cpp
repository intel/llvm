//==--------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/ext/oneapi/experimental/graph.hpp"
#include <sycl/sycl.hpp>

#include "../thread_safety/ThreadUtils.h"
#include "detail/graph_impl.hpp"

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

using namespace sycl;
using namespace sycl::ext::oneapi;

// anonymous namespace used to avoid code redundancy by defining functions
// used by multiple times by unitests.
// Defining anonymous namespace prevents from function naming conflits
namespace {
/// Submits four kernels with diamond dependency to the queue Q
/// @param Q Queue to submit nodes to.
void runKernels(queue Q) {
  auto NodeA = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(NodeA);
    cgh.single_task<TestKernel<>>([]() {});
  });
  auto NodeC = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(NodeA);
    cgh.single_task<TestKernel<>>([]() {});
  });
  auto NodeD = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({NodeB, NodeC});
    cgh.single_task<TestKernel<>>([]() {});
  });
}

/// Submits four kernels without any additional dependencies the queue Q
/// @param Q Queue to submit nodes to.
void runKernelsInOrder(queue Q) {
  auto NodeA = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeC = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeD = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
}

/// Adds four kernels with diamond dependency to the Graph G
/// @param G Modifiable graph to add commands to.
void addKernels(
    experimental::command_graph<experimental::graph_state::modifiable> G) {
  auto NodeA = G.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeA)});
  auto NodeC =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeA)});
  auto NodeD =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeB, NodeC)});
}

bool checkExecGraphSchedule(
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphA,
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphB) {
  auto ScheduleA = GraphA->getSchedule();
  auto ScheduleB = GraphB->getSchedule();
  if (ScheduleA.size() != ScheduleB.size())
    return false;

  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleA{std::begin(ScheduleA), std::end(ScheduleA)};
  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleB{std::begin(ScheduleB), std::end(ScheduleB)};

  for (size_t i = 0; i < VScheduleA.size(); i++) {
    if (!VScheduleA[i]->isSimilar(VScheduleB[i]))
      return false;
  }
  return true;
}

/// Define the three possible path to add node to a SYCL Graph.
/// Shortcut is a sub-type of Record&Replay using Queue shortcut
/// instead of standard kernel submitions.
enum OperationPath { Explicit, RecordReplay, Shortcut };

/// Tries to add a memcpy2D node to the graph G
/// It tests that an invalid exception has been thrown
/// Since sycl_ext_oneapi_memcpy2d extension can not be used
/// along with SYCL Graph.
///
/// @param G Modifiable graph to add commands to.
/// @param Q Queue to submit nodes to.
/// @param Dest Pointer to the memory destination
/// @param DestPitch pitch at the destination
/// @param Src Pointer to the memory source
/// @param SrcPitch pitch at the source
/// @param Witdh width of the data to copy
/// @param Height height of the data to copy
template <OperationPath PathKind>
void addMemcpy2D(experimental::detail::modifiable_command_graph &G, queue &Q,
                 void *Dest, size_t DestPitch, const void *Src, size_t SrcPitch,
                 size_t Width, size_t Height) {
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}

/// Tries to add nodes including images bindless copy instructions
/// to the graph G. It tests that an invalid exception has been thrown
/// Since sycl_ext_oneapi_bindless_images extension can not be used
/// along with SYCL Graph.
///
/// @param G Modifiable graph to add commands to.
/// @param Q Queue to submit nodes to.
/// @param Img Image memory
/// @param HostData Host Pointer to the memory
/// @param ImgUSM USM Pointer to Image memory
/// @param Pitch image pitch
/// @param Desc Image descriptor
template <OperationPath PathKind>
void addImagesCopies(experimental::detail::modifiable_command_graph &G,
                     queue &Q, sycl::ext::oneapi::experimental::image_mem Img,
                     std::vector<sycl::float4> HostData, void *ImgUSM,
                     size_t Pitch,
                     sycl::ext::oneapi::experimental::image_descriptor Desc) {
  // simple copy Host to Device
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), Img.get_handle(), Desc);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(HostData.data(), Img.get_handle(), Desc);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), Img.get_handle(), Desc);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  // simple copy Device to Host
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(Img.get_handle(), HostData.data(), Desc);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(Img.get_handle(), HostData.data(), Desc);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(Img.get_handle(), HostData.data(), Desc);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  // simple copy Host to Device USM
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), ImgUSM, Desc, Pitch);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(HostData.data(), ImgUSM, Desc, Pitch);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), ImgUSM, Desc, Pitch);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  // subregion copy Host to Device
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), {0, 0, 0}, {0, 0, 0},
                            Img.get_handle(), {0, 0, 0}, Desc, {0, 0, 0});
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(HostData.data(), {0, 0, 0}, {0, 0, 0}, Img.get_handle(),
                        {0, 0, 0}, Desc, {0, 0, 0});
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), {0, 0, 0}, {0, 0, 0},
                            Img.get_handle(), {0, 0, 0}, Desc, {0, 0, 0});
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  // subregion copy Device to Host
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(Img.get_handle(), {0, 0, 0}, Desc, HostData.data(),
                            {0, 0, 0}, {0, 0, 0}, {0, 0, 0});
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(Img.get_handle(), {0, 0, 0}, Desc, HostData.data(),
                        {0, 0, 0}, {0, 0, 0}, {0, 0, 0});
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(Img.get_handle(), {0, 0, 0}, Desc, HostData.data(),
                            {0, 0, 0}, {0, 0, 0}, {0, 0, 0});
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  // subregion copy Host to Device USM
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), {0, 0, 0}, ImgUSM, {0, 0, 0}, Desc,
                            Pitch, {0, 0, 0}, {0, 0, 0});
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.ext_oneapi_copy(HostData.data(), {0, 0, 0}, ImgUSM, {0, 0, 0}, Desc,
                        Pitch, {0, 0, 0}, {0, 0, 0});
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.ext_oneapi_copy(HostData.data(), {0, 0, 0}, ImgUSM, {0, 0, 0}, Desc,
                            Pitch, {0, 0, 0}, {0, 0, 0});
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}
} // anonymous namespace

class CommandGraphTest : public ::testing::Test {
public:
  CommandGraphTest()
      : Mock{}, Plat{Mock.getPlatform()}, Dev{Plat.get_devices()[0]},
        Queue{Dev},
        Graph{Queue.get_context(),
              Dev,
              {experimental::property::graph::assume_buffer_outlives_graph{}}} {
  }

protected:
  void SetUp() override {}

protected:
  unittest::PiMock Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};

TEST_F(CommandGraphTest, QueueState) {
  experimental::queue_state State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::executing);

  experimental::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);
  State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::recording);

  Graph.end_recording();
  State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::executing);
}

TEST_F(CommandGraphTest, AddNode) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl->MRoots.empty());

  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node1), nullptr);
  ASSERT_FALSE(sycl::detail::getSyclObjImpl(Node1)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ((*GraphImpl->MRoots.begin()).lock(),
            sycl::detail::getSyclObjImpl(Node1));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());

  // Add a node which depends on the first
  auto Node2Deps = experimental::property::node::depends_on(Node1);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2Deps.get_dependencies().front()),
            sycl::detail::getSyclObjImpl(Node1));
  auto Node2 = Graph.add([&](sycl::handler &cgh) {}, {Node2Deps});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front().lock(),
            sycl::detail::getSyclObjImpl(Node2));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);

  // Add a third node which depends on both
  auto Node3 =
      Graph.add([&](sycl::handler &cgh) {},
                {experimental::property::node::depends_on(Node1, Node2)});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node3), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node3)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);

  // Add a fourth node without any dependencies on the others
  auto Node4 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node4), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node4)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node3)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node4)->MPredecessors.empty());
}

TEST_F(CommandGraphTest, Finalize) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  sycl::buffer<int> Buf(1);
  auto Node1 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::write_only, sycl::no_init);
    cgh.single_task<TestKernel<>>([]() {});
  });

  // Add independent node
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Add a node that depends on Node1 due to the accessor
  auto Node3 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::write_only, sycl::no_init);
    cgh.single_task<TestKernel<>>([]() {});
  });

  // Guarantee order of independent nodes 1 and 2
  Graph.make_edge(Node2, Node1);

  auto GraphExec = Graph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);

  // The final schedule should contain three nodes in order: 2->1->3
  auto Schedule = GraphExecImpl->getSchedule();
  ASSERT_EQ(Schedule.size(), 3ul);
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node2));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3));
  ASSERT_EQ(Queue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, MakeEdge) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Add two independent nodes
  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_EQ(GraphImpl->MRoots.size(), 2ul);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.empty());

  // Connect nodes and verify order
  Graph.make_edge(Node1, Node2);
  ASSERT_EQ(GraphImpl->MRoots.size(), 1ul);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front().lock(),
            sycl::detail::getSyclObjImpl(Node2));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
}

TEST_F(CommandGraphTest, BeginEndRecording) {
  sycl::queue Queue2{Queue.get_context(), Dev};

  // Test throwing behaviour
  // Check we can repeatedly begin recording on the same queues
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  // Check we can repeatedly end recording on the same queues
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  // Vector versions
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));

  experimental::command_graph Graph2(Queue.get_context(), Dev);

  Graph.begin_recording(Queue);
  // Trying to record to a second Graph should throw
  ASSERT_ANY_THROW(Graph2.begin_recording(Queue));
  // Trying to end when it is recording to a different graph should throw
  ASSERT_ANY_THROW(Graph2.end_recording(Queue));
  Graph.end_recording(Queue);

  // Testing return values of begin and end recording
  // Queue should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue));

  // Queue2 should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue2));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // Queue should have changed state so should return true
  ASSERT_TRUE(Graph.end_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.end_recording(Queue));

  // Should end recording on Queue2
  ASSERT_TRUE(Graph.end_recording());
  // State should not change on Queue2 now
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // Testing vector begin and end
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.begin_recording(Queue));
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // End recording on both
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.end_recording(Queue));
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // First add one single queue
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // Vector begin should still return true as Queue2 has state changed
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // End recording on Queue2
  ASSERT_TRUE(Graph.end_recording(Queue2));
  // Vector end should still return true as Queue will have state changed
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
}

TEST_F(CommandGraphTest, GetCGCopy) {
  auto Node1 = Graph.add([&](sycl::handler &cgh) {});
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1)});

  // Get copy of CG of Node2 and check equality
  auto Node2Imp = sycl::detail::getSyclObjImpl(Node2);
  auto Node2CGCopy = Node2Imp->getCGCopy();
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp->MCGType);
  ASSERT_EQ(Node2CGCopy->getType(), sycl::detail::CG::Kernel);
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp->MCommandGroup->getType());
  ASSERT_EQ(Node2CGCopy->getAccStorage(),
            Node2Imp->MCommandGroup->getAccStorage());
  ASSERT_EQ(Node2CGCopy->getArgsStorage(),
            Node2Imp->MCommandGroup->getArgsStorage());
  ASSERT_EQ(Node2CGCopy->getEvents(), Node2Imp->MCommandGroup->getEvents());
  ASSERT_EQ(Node2CGCopy->getRequirements(),
            Node2Imp->MCommandGroup->getRequirements());
  ASSERT_EQ(Node2CGCopy->getSharedPtrStorage(),
            Node2Imp->MCommandGroup->getSharedPtrStorage());
}
TEST_F(CommandGraphTest, SubGraph) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->isEmpty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph.
  // we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 5 nodes: 4 regulars + 1 empty.
  // Indeed an empty node is added as an exit point of the added subgraph to
  // facilitate the handling of dependencies
  ASSERT_EQ(Schedule.size(), 5ul);
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1MainGraph));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, SubGraphWithEmptyNode) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Empty1Graph =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on(Node1Graph)});
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Empty1Graph)});

  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->isEmpty());
  // Check the structure of the main graph.
  // 1 root connected to 1 successor (the single root of the subgraph)
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph.
  // we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 6 nodes: 5 regulars + 1 empty.
  // Indeed an empty node is added as an exit point of the added subgraph to
  // facilitate the handling of dependencies
  ASSERT_EQ(Schedule.size(), 6ul);
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1MainGraph));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty()); // empty node inside the subgraph
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE(
      (*ScheduleIt)->isEmpty()); // empty node added by the impl to handle
                                 // depenendcies w.r.t. the added subgraph
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, SubGraphWithEmptyNodeLast) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto EmptyGraph =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on(Node2Graph)});

  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->isEmpty());
  // Check the structure of the main graph.
  // 1 root connected to 1 successor (the single root of the subgraph)
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph.
  // we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 6 nodes: 5 regulars + 1 empty.
  // Indeed an empty node is added as an exit point of the added subgraph to
  // facilitate the handling of dependencies
  ASSERT_EQ(Schedule.size(), 6ul);
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1MainGraph));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty()); // empty node inside the subgraph
  ScheduleIt++;
  ASSERT_TRUE(
      (*ScheduleIt)->isEmpty()); // empty node added by the impl to handle
                                 // depenendcies w.r.t. the added subgraph
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, RecordSubGraph) {
  // Record sub-graph with two nodes
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node1Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });
  Graph.end_recording(Queue);
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  MainGraph.begin_recording(Queue);
  auto Node1MainGraph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph = Queue.submit([&](handler &cgh) {
    cgh.depends_on(Node1MainGraph);
    cgh.ext_oneapi_graph(GraphExec);
  });
  auto Node3MainGraph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node2MainGraph);
    cgh.single_task<TestKernel<>>([]() {});
  });
  MainGraph.end_recording(Queue);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 5 nodes: 4 regulars + 1 empty.
  // Indeed an empty node is added as an exit point of the added subgraph to
  // facilitate the handling of dependencies
  ASSERT_EQ(Schedule.size(), 5ul);

  // The first and fourth nodes should have events associated with MainGraph but
  // not graph. The second and third nodes were added as a sub-graph and
  // duplicated. They should not have events associated with Graph or MainGraph.
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt),
      sycl::detail::getSyclObjImpl(Node1MainGraph));

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt));
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt));
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));

  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt),
      sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmpty) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with a regular node then empty node then a regular
  // node
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyFirst) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with an empty node then two regular nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyLast) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with two regular nodes then an empty node
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithPreviousHostTask) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  auto EventInitial =
      InOrderQueue.submit([&](handler &CGH) { CGH.host_task([=]() {}); });
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto EventLast = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto WaitList = EventLastImpl->getWaitList();
  // Previous task is a host task. Explicit dependency is needed to enforce the
  // execution order
  ASSERT_EQ(WaitList.size(), 1lu);
  ASSERT_EQ(WaitList[0], EventInitialImpl);
}

TEST_F(CommandGraphTest, InOrderQueueHostTaskAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  auto EventInitial =
      InOrderQueue.submit([&](handler &CGH) { CGH.host_task([=]() {}); });
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is an host task. Explicit dependency is needed to enfore the
  // execution order
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not an host task. In Order queue dependency are managed by
  // the backend for non-host kernels
  ASSERT_EQ(EventLastWaitList.size(), 0lu);
}

TEST_F(CommandGraphTest, InOrderQueueMemsetAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Check if device has usm shared allocation
  if (!InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations))
    return;
  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial = InOrderQueue.memset(TestData, 1, Size * sizeof(int));
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is an host task. Explicit dependency is needed to enfore the
  // execution order
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not a host task. In Order queue dependency is managed by
  // the backend for non-host kernels.
  ASSERT_EQ(EventLastWaitList.size(), 0lu);
}

TEST_F(CommandGraphTest, InOrderQueueMemcpyAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Check if device has usm shared allocation.
  if (!InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations))
    return;
  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is an host task. Explicit dependency is needed to enfore the
  // execution order
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not an host task. In Order queue dependency are managed by
  // the backend for non-host kernels
  ASSERT_EQ(EventLastWaitList.size(), 0lu);
}

TEST_F(CommandGraphTest, ExplicitBarrierException) {

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    auto Barrier =
        Graph.add([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}

TEST_F(CommandGraphTest, EnqueueBarrier) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier =
      Queue.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (B)
  //     / \
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto BarrierNode = Node->MSuccessors.front().lock();
    ASSERT_EQ(BarrierNode->MCGType, sycl::detail::CG::Barrier);
    ASSERT_EQ(GraphImpl->getEventForNode(BarrierNode),
              sycl::detail::getSyclObjImpl(Barrier));
    ASSERT_EQ(BarrierNode->MPredecessors.size(), 3lu);
    ASSERT_EQ(BarrierNode->MSuccessors.size(), 2lu);
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierMultipleQueues) {
  sycl::queue Queue2{Queue.get_context(), Dev};
  Graph.begin_recording({Queue, Queue2});
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier =
      Queue2.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node4Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (B)
  //     / \
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto BarrierNode = Node->MSuccessors.front().lock();
    ASSERT_EQ(BarrierNode->MCGType, sycl::detail::CG::Barrier);
    ASSERT_EQ(GraphImpl->getEventForNode(BarrierNode),
              sycl::detail::getSyclObjImpl(Barrier));
    ASSERT_EQ(BarrierNode->MPredecessors.size(), 3lu);
    ASSERT_EQ(BarrierNode->MSuccessors.size(), 2lu);
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierWaitList) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //     (B)  |
  //     / \ /
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 2lu);
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
    }
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierWaitListMultipleQueues) {
  sycl::queue Queue2{Queue.get_context(), Dev};
  Graph.begin_recording({Queue, Queue2});
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Node1Graph comes from Queue, and Node2Graph comes from Queue2
  auto Barrier = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  auto Barrier2 = Queue2.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Barrier, Node4Graph, Node5Graph});
  });

  Graph.end_recording();

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //     (B)  |
  //     /|\ /
  //   (4)|(5)
  //    \ | /
  //     \|/
  //     (B2)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 3lu);
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
    }
  }
}

TEST_F(CommandGraphTest, EnqueueMultipleBarrier) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier1 = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  auto Barrier2 =
      Queue.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node6Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node7Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node8Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //    (B1)  |
  //     /|\ /
  //   (4)|(5)
  //     \|/
  //     (B2)
  //     /|\
  //    / | \
  // (6) (7) (8)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier1));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 3lu);
      for (auto Succ1 : SuccNode->MSuccessors) {
        auto SuccBarrier1 = Succ1.lock();
        if (SuccBarrier1->MCGType == sycl::detail::CG::Barrier) {
          ASSERT_EQ(GraphImpl->getEventForNode(SuccBarrier1),
                    sycl::detail::getSyclObjImpl(Barrier2));
          ASSERT_EQ(SuccBarrier1->MPredecessors.size(), 3lu);
          ASSERT_EQ(SuccBarrier1->MSuccessors.size(), 3lu);
          for (auto Succ2 : SuccBarrier1->MSuccessors) {
            auto SuccBarrier2 = Succ2.lock();
            // Nodes 6, 7, 8
            ASSERT_EQ(SuccBarrier2->MPredecessors.size(), 1lu);
            ASSERT_EQ(SuccBarrier2->MSuccessors.size(), 0lu);
          }
        } else {
          // Node 4 or Node 5
          if (GraphImpl->getEventForNode(SuccBarrier1) ==
              sycl::detail::getSyclObjImpl(Node4Graph)) {
            // Node 4
            ASSERT_EQ(SuccBarrier1->MPredecessors.size(), 1lu);
            ASSERT_EQ(SuccBarrier1->MSuccessors.size(), 1lu);
          }
        }
      }
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 1lu);
    }
  }
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword1) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (E)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 0lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MSuccessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword2) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node3Graph)});

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | (4)
  //     \| /
  //     (E)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 0lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);

  auto Node4Impl = sycl::detail::getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MSuccessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword3) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(EmptyNode)});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //  |\ |
  //  | (E)
  // (3) |
  //    (4)
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 1lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 2lu);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);

  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MPredecessors[0].lock(), Node1Impl);

  auto Node4Impl = sycl::detail::getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MPredecessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword4) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode2 =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //   \/
  //  (E1) (3)
  //    \  /
  //    (E2)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 1lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);

  auto EmptyImpl2 = sycl::detail::getSyclObjImpl(EmptyNode2);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MPredecessors.size(), 0lu);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MSuccessors[0].lock(), EmptyImpl2);

  ASSERT_EQ(EmptyImpl2->MPredecessors.size(), 2lu);
}

TEST_F(CommandGraphTest, FusionExtensionExceptionCheck) {
  device D;
  if (!D.get_info<
          ext::codeplay::experimental::info::device::supports_fusion>()) {
    // Skip this test if the device does not support fusion. Otherwise, the
    // queue construction in the next step would fail.
    GTEST_SKIP();
  }

  queue Q{D, ext::codeplay::experimental::property::queue::enable_fusion{}};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Q.get_context(), Q.get_device()};

  ext::codeplay::experimental::fusion_wrapper fw{Q};

  // Test: Start fusion on a queue that is in recording mode
  Graph.begin_recording(Q);

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    fw.start_fusion();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  Graph.end_recording(Q);

  // Test: begin recording a queue in fusion mode

  fw.start_fusion();

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.begin_recording(Q);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}

TEST_F(CommandGraphTest, USMMemsetShortcutExceptionCheck) {

  const size_t N = 10;
  unsigned char *Arr = malloc_device<unsigned char>(N, Queue);
  int Value = 77;

  Graph.begin_recording(Queue);

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Queue.memset(Arr, Value, N);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  Graph.end_recording(Queue);
}

TEST_F(CommandGraphTest, Memcpy2DExceptionCheck) {
  constexpr size_t RECT_WIDTH = 30;
  constexpr size_t RECT_HEIGHT = 21;
  constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
  constexpr size_t DST_ELEMS = SRC_ELEMS;

  using T = int;

  Graph.begin_recording(Queue);

  T *USMMemSrc = malloc_device<T>(SRC_ELEMS, Queue);
  T *USMMemDst = malloc_device<T>(DST_ELEMS, Queue);

  addMemcpy2D<OperationPath::RecordReplay>(
      Graph, Queue, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
      RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);

  addMemcpy2D<OperationPath::Shortcut>(
      Graph, Queue, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
      RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);

  Graph.end_recording();

  addMemcpy2D<OperationPath::Explicit>(
      Graph, Queue, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
      RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);

  sycl::free(USMMemSrc, Queue);
  sycl::free(USMMemDst, Queue);
}

// Tests that using reductions in a graph will throw.
TEST_F(CommandGraphTest, Reductions) {
  int ReduVar = 0;
  ASSERT_THROW(
      {
        try {
          Graph.add([&](handler &CGH) {
            CGH.parallel_for<class CustomTestKernel>(
                range<1>{1}, reduction(&ReduVar, int{0}, sycl::plus<>()),
                [=](item<1> idx, auto &Sum) {});
          });
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);
}

TEST_F(CommandGraphTest, BindlessExceptionCheck) {
  auto Ctxt = Queue.get_context();

  // declare image data
  size_t Height = 13;
  size_t Width = 7;
  size_t Depth = 11;
  size_t N = Height * Width * Depth;
  std::vector<sycl::float4> DataIn(N);

  // Extension: image descriptor - can use the same for both images
  sycl::ext::oneapi::experimental::image_descriptor Desc(
      {Width, Height, Depth}, sycl::image_channel_order::rgba,
      sycl::image_channel_type::fp32);

  // Extension: allocate memory on device and create the handle
  // Input images memory
  sycl::ext::oneapi::experimental::image_mem ImgMem(Desc, Dev, Ctxt);
  // Extension: returns the device pointer to USM allocated pitched memory
  size_t Pitch = 0;
  auto ImgMemUSM = sycl::ext::oneapi::experimental::pitched_alloc_device(
      &Pitch, Desc, Queue);

  Graph.begin_recording(Queue);

  addImagesCopies<OperationPath::RecordReplay>(Graph, Queue, ImgMem, DataIn,
                                               ImgMemUSM, Pitch, Desc);

  addImagesCopies<OperationPath::Shortcut>(Graph, Queue, ImgMem, DataIn,
                                           ImgMemUSM, Pitch, Desc);

  Graph.end_recording();

  addImagesCopies<OperationPath::Explicit>(Graph, Queue, ImgMem, DataIn,
                                           ImgMemUSM, Pitch, Desc);

  sycl::free(ImgMemUSM, Ctxt);
}

TEST_F(CommandGraphTest, GetProfilingInfoExceptionCheck) {
  sycl::context Ctx{Dev};
  sycl::queue QueueProfile{
      Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  experimental::command_graph<experimental::graph_state::modifiable>
      GraphProfile{QueueProfile.get_context(), Dev};

  GraphProfile.begin_recording(QueueProfile);
  auto Event = QueueProfile.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Checks that exception is thrown when get_profile_info is called on "event"
  // returned by a queue in recording mode.
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Event.get_profiling_info<sycl::info::event_profiling::command_end>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  GraphProfile.end_recording();

  auto GraphExec = GraphProfile.finalize();
  auto EventSub = QueueProfile.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  // Checks that exception is thrown when get_profile_info is called on "event"
  // returned by a graph submission.
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    EventSub.get_profiling_info<sycl::info::event_profiling::command_submit>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    EventSub.get_profiling_info<sycl::info::event_profiling::command_start>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    EventSub.get_profiling_info<sycl::info::event_profiling::command_end>();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}

TEST_F(CommandGraphTest, MakeEdgeErrors) {
  // Set up some nodes in the graph
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Test error on calling make_edge when a queue is recording to the graph
  Graph.begin_recording(Queue);
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(NodeA, NodeB);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  Graph.end_recording(Queue);

  // Test error on Src and Dest being the same
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(NodeA, NodeA);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  // Test Src or Dest not being found in the graph
  experimental::command_graph<experimental::graph_state::modifiable> GraphOther{
      Queue.get_context(), Queue.get_device()};
  auto NodeOther = GraphOther.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  ASSERT_THROW(
      {
        try {
          Graph.make_edge(NodeA, NodeOther);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(NodeOther, NodeB);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  // Test that adding a cycle with cycle checks leaves the graph in the correct
  // state.

  auto CheckGraphStructure = [&]() {
    auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
    auto NodeAImpl = sycl::detail::getSyclObjImpl(NodeA);
    auto NodeBImpl = sycl::detail::getSyclObjImpl(NodeB);

    ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
    ASSERT_EQ((*GraphImpl->MRoots.begin()).lock(), NodeAImpl);

    ASSERT_EQ(NodeAImpl->MSuccessors.size(), 1lu);
    ASSERT_EQ(NodeAImpl->MPredecessors.size(), 0lu);
    ASSERT_EQ(NodeAImpl->MSuccessors.front().lock(), NodeBImpl);

    ASSERT_EQ(NodeBImpl->MSuccessors.size(), 0lu);
    ASSERT_EQ(NodeBImpl->MPredecessors.size(), 1lu);
    ASSERT_EQ(NodeBImpl->MPredecessors.front().lock(), NodeAImpl);
  };
  // Make a normal edge
  ASSERT_NO_THROW(Graph.make_edge(NodeA, NodeB));

  // Check the expected structure of the graph
  CheckGraphStructure();

  // Introduce a cycle, make sure it throws
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(NodeB, NodeA);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  // Re-check graph structure to make sure the graph state has not been modified
  CheckGraphStructure();
}

TEST_F(CommandGraphTest, InvalidBuffer) {
  // Check that using a buffer with write_back enabled in a graph will throw.
  int Data;
  // Create a buffer which does not have write-back disabled.
  buffer<int> Buffer{&Data, range<1>{1}};

  // Use this buffer in the graph, this should throw.
  ASSERT_THROW(
      {
        try {
          Graph.add([&](handler &CGH) {
            auto Acc = Buffer.get_access<access::mode::read_write>(CGH);
          });
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);
}

TEST_F(CommandGraphTest, InvalidHostAccessor) {
  // Check that creating a host_accessor on a buffer which is in use by a graph
  // will throw.

  // Create a buffer which does not have write-back disabled.
  buffer<int> Buffer{range<1>{1}};

  {
    // Create a graph in local scope so we can destroy it
    ext::oneapi::experimental::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {experimental::property::graph::assume_buffer_outlives_graph{}}};

    // Add the buffer to the graph.
    Graph.add([&](handler &CGH) {
      auto Acc = Buffer.get_access<access::mode::read_write>(CGH);
    });

    // Attempt to create a host_accessor, which should throw.
    ASSERT_THROW(
        {
          try {
            host_accessor HostAcc{Buffer};
          } catch (const sycl::exception &e) {
            ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
            throw;
          }
        },
        sycl::exception);
  }
  // Graph is now out of scope so we should be able to create a host_accessor
  ASSERT_NO_THROW({ host_accessor HostAcc{Buffer}; });
}

TEST_F(CommandGraphTest, GraphPartitionsMerging) {
  // Tests that the parition merging algo works as expected in case of backward
  // dependencies
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeA)});
  auto NodeHT1 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeB)});
  auto NodeC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT1)});
  auto NodeD = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeB)});
  auto NodeHT2 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeD)});
  auto NodeE = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});
  auto NodeF = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});

  // Backward dependency
  Graph.make_edge(NodeE, NodeHT1);

  auto GraphExec = Graph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto PartitionsList = GraphExecImpl->getPartitions();
  ASSERT_EQ(PartitionsList.size(), 5ul);
  ASSERT_FALSE(PartitionsList[0]->isHostTask());
  ASSERT_TRUE(PartitionsList[1]->isHostTask());
  ASSERT_FALSE(PartitionsList[2]->isHostTask());
  ASSERT_TRUE(PartitionsList[3]->isHostTask());
  ASSERT_FALSE(PartitionsList[4]->isHostTask());
}

TEST_F(CommandGraphTest, ProfilingException) {
  Graph.begin_recording(Queue);
  auto Event1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Event2 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording(Queue);

  try {
    Event1.get_profiling_info<sycl::info::event_profiling::command_start>();
  } catch (exception &Exception) {
    ASSERT_FALSE(
        std::string(Exception.what())
            .find("Profiling information is unavailable for events returned "
                  "from a submission to a queue in the recording state.") ==
        std::string::npos);
  }
}

class MultiThreadGraphTest : public CommandGraphTest {
public:
  MultiThreadGraphTest()
      : CommandGraphTest(), NumThreads(std::thread::hardware_concurrency()),
        SyncPoint(NumThreads) {
    Threads.reserve(NumThreads);
  }

protected:
  const unsigned NumThreads;
  Barrier SyncPoint;
  std::vector<std::thread> Threads;
};

TEST_F(MultiThreadGraphTest, BeginEndRecording) {
  auto RecordGraph = [&]() {
    queue MyQueue{Queue.get_context(), Queue.get_device()};

    SyncPoint.wait();

    Graph.begin_recording(MyQueue);
    runKernels(MyQueue);
    Graph.end_recording(MyQueue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef{Queue.get_context(), Queue.get_device()};
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  for (unsigned i = 0; i < NumThreads; ++i) {
    queue MyQueue{Queue.get_context(), Queue.get_device()};
    GraphRef.begin_recording(MyQueue);
    runKernels(MyQueue);
    GraphRef.end_recording(MyQueue);
  }

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, ExplicitAddNodes) {
  auto RecordGraph = [&]() {
    queue MyQueue{Queue.get_context(), Queue.get_device()};

    SyncPoint.wait();
    addKernels(Graph);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef;
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  for (unsigned i = 0; i < NumThreads; ++i) {
    addKernels(GraphRef);
  }

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, RecordAddNodes) {
  Graph.begin_recording(Queue);
  auto RecordGraph = [&]() {
    SyncPoint.wait();
    runKernels(Queue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // We stop recording the Queue when all threads have finished their processing
  Graph.end_recording(Queue);

  // Reference computation
  queue QueueRef{Queue.get_context(), Queue.get_device()};
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  GraphRef.begin_recording(QueueRef);
  for (unsigned i = 0; i < NumThreads; ++i) {
    runKernels(QueueRef);
  }
  GraphRef.end_recording(QueueRef);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, RecordAddNodesInOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  queue InOrderQueue{Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  InOrderGraph.begin_recording(InOrderQueue);
  auto RecordGraph = [&]() {
    SyncPoint.wait();
    runKernelsInOrder(InOrderQueue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // We stop recording the Queue when all threads have finished their processing
  InOrderGraph.end_recording(InOrderQueue);

  // Reference computation
  queue InOrderQueueRef{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraphRef{InOrderQueueRef.get_context(),
                      InOrderQueueRef.get_device()};

  InOrderGraphRef.begin_recording(InOrderQueueRef);
  for (unsigned i = 0; i < NumThreads; ++i) {
    runKernelsInOrder(InOrderQueueRef);
  }
  InOrderGraphRef.end_recording(InOrderQueueRef);

  auto GraphImpl = sycl::detail::getSyclObjImpl(InOrderGraph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(InOrderGraphRef);
  ASSERT_EQ(GraphImpl->getNumberOfNodes(), GraphRefImpl->getNumberOfNodes());

  // In-order graph must have only a single root
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);

  // Check structure graph
  auto CurrentNode = (*GraphImpl->MRoots.begin()).lock();
  for (size_t i = 1; i <= GraphImpl->getNumberOfNodes(); i++) {
    EXPECT_LE(CurrentNode->MSuccessors.size(), 1lu);

    // Checking the last node has no successors
    if (i == GraphImpl->getNumberOfNodes()) {
      EXPECT_EQ(CurrentNode->MSuccessors.size(), 0lu);
    } else {
      // Check other nodes have 1 successor
      EXPECT_EQ(CurrentNode->MSuccessors.size(), 1lu);
      CurrentNode = CurrentNode->MSuccessors[0].lock();
    }
  }
}

TEST_F(MultiThreadGraphTest, Finalize) {
  addKernels(Graph);

  std::mutex MutexMap;

  std::map<int,
           experimental::command_graph<experimental::graph_state::executable>>
      GraphsExecMap;
  auto FinalizeGraph = [&](int ThreadNum) {
    SyncPoint.wait();
    auto GraphExec = Graph.finalize();
    Queue.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

    std::lock_guard<std::mutex> Guard(MutexMap);
    GraphsExecMap.insert(
        std::map<int, experimental::command_graph<
                          experimental::graph_state::executable>>::
            value_type(ThreadNum, GraphExec));
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(FinalizeGraph, i);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef;
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  addKernels(GraphRef);

  for (unsigned i = 0; i < NumThreads; ++i) {
    auto GraphExecRef = GraphRef.finalize();
    QueueRef.submit(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExecRef); });
    auto GraphExecImpl =
        sycl::detail::getSyclObjImpl(GraphsExecMap.find(i)->second);
    auto GraphExecRefImpl = sycl::detail::getSyclObjImpl(GraphExecRef);
    ASSERT_EQ(checkExecGraphSchedule(GraphExecImpl, GraphExecRefImpl), true);
  }
}

// Test adding fill and memset nodes to a graph
TEST_F(CommandGraphTest, FillMemsetNodes) {
  const int Value = 7;
  // Buffer fill
  buffer<int> Buffer{range<1>{1}};
  Buffer.set_write_back(false);

  {
    ext::oneapi::experimental::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {experimental::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeA = Graph.add([&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Value);
    });
    auto NodeB = Graph.add([&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Value);
    });

    auto NodeAImpl = sycl::detail::getSyclObjImpl(NodeA);
    auto NodeBImpl = sycl::detail::getSyclObjImpl(NodeB);

    // Check Operator==
    EXPECT_EQ(NodeAImpl, NodeAImpl);
    EXPECT_NE(NodeAImpl, NodeBImpl);
  }

  // USM
  {
    int *USMPtr = malloc_device<int>(1, Queue);

    // We need to create some differences between nodes because unlike buffer
    // fills they are not differentiated on accessor ptr value.
    auto FillNodeA =
        Graph.add([&](handler &CGH) { CGH.fill(USMPtr, Value, 1); });
    auto FillNodeB =
        Graph.add([&](handler &CGH) { CGH.fill(USMPtr, Value + 1, 1); });
    auto MemsetNodeA =
        Graph.add([&](handler &CGH) { CGH.memset(USMPtr, Value, 1); });
    auto MemsetNodeB =
        Graph.add([&](handler &CGH) { CGH.memset(USMPtr, Value, 2); });

    auto FillNodeAImpl = sycl::detail::getSyclObjImpl(FillNodeA);
    auto FillNodeBImpl = sycl::detail::getSyclObjImpl(FillNodeB);
    auto MemsetNodeAImpl = sycl::detail::getSyclObjImpl(MemsetNodeA);
    auto MemsetNodeBImpl = sycl::detail::getSyclObjImpl(MemsetNodeB);

    // Check Operator==
    EXPECT_EQ(FillNodeAImpl, FillNodeAImpl);
    EXPECT_EQ(FillNodeBImpl, FillNodeBImpl);
    EXPECT_NE(FillNodeAImpl, FillNodeBImpl);

    EXPECT_EQ(MemsetNodeAImpl, MemsetNodeAImpl);
    EXPECT_EQ(MemsetNodeBImpl, MemsetNodeBImpl);
    EXPECT_NE(MemsetNodeAImpl, MemsetNodeBImpl);
    sycl::free(USMPtr, Queue);
  }
}
