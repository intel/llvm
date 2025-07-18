//==------------------------- Exceptions.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

// Various tests which are checking for correct exception behaviour

// anonymous namespace used to avoid code redundancy by defining functions
// used by multiple times by unitests.
// Defining anonymous namespace prevents from function naming conflits
namespace {
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

/// Tries to add nodes including asynchronous allocation instructions to the
/// graph G. It tests that an invalid exception has been thrown since the
/// sycl_ext_oneapi_async_alloc can not be used along with SYCL Graph.
///
/// @param G Modifiable graph to add commands to.
/// @param Q Queue to submit nodes to.
/// @param Size Size in bytes to allocate.
/// @param Ptr Generic pointer to allocated memory.
template <OperationPath PathKind, usm::alloc AllocKind>
void addAsyncAlloc(experimental::detail::modifiable_command_graph &G, queue &Q,
                   size_t Size, [[maybe_unused]] void *Ptr) {
  // simple alloc
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        Ptr =
            sycl::ext::oneapi::experimental::async_malloc(CGH, AllocKind, Size);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Ptr = sycl::ext::oneapi::experimental::async_malloc(Q, AllocKind, Size);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        Ptr =
            sycl::ext::oneapi::experimental::async_malloc(CGH, AllocKind, Size);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }

  ASSERT_EQ(ExceptionCode, sycl::errc::feature_not_supported);
}
} // anonymous namespace

TEST_F(CommandGraphTest, ExplicitBarrierException) {
  bool Success = true;
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    auto Barrier =
        Graph.add([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
    std::string ErrorStr =
        "The sycl_ext_oneapi_enqueue_barrier feature is "
        "not available with SYCL Graph Explicit API. Please use empty nodes "
        "instead.";
    std::cout << Exception.what() << std::endl;
    std::cout << ErrorStr << std::endl;
    ASSERT_FALSE(std::string(Exception.what()).find(ErrorStr) ==
                 std::string::npos);
    Success = false;
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
  ASSERT_EQ(Success, false);
}

TEST_F(CommandGraphTest, ExplicitBarrierDependencyException) {

  experimental::command_graph<experimental::graph_state::modifiable> Graph2{
      Queue};

  Graph2.begin_recording({Queue});

  auto Node = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Graph2.end_recording();

  auto Event = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  Graph.begin_recording(Queue);

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    auto BarrierNode = Queue.ext_oneapi_submit_barrier({Node});
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    auto BarrierNode = Queue.ext_oneapi_submit_barrier({Event});
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);

  Graph2.end_recording();
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

// Test that using sycl streams in a graph node will throw
TEST_F(CommandGraphTest, Streams) {
  ASSERT_THROW(
      {
        size_t WorkItems = 16;
        try {
          Graph.add([&](handler &CGH) {
            sycl::stream Out(WorkItems * 16, 16, CGH);
            CGH.parallel_for<class CustomTestKernel>(
                range<1>(WorkItems),
                [=](item<1> id) { Out << id.get_linear_id() << sycl::endl; });
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
      {Width, Height, Depth}, 4, sycl::image_channel_type::fp32);

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

// sycl_ext_oneapi_work_group_scratch_memory isn't supported with SYCL graphs
TEST_F(CommandGraphTest, WorkGroupScratchMemoryCheck) {
  ASSERT_THROW(
      {
        try {
          Graph.add([&](handler &CGH) {
            CGH.parallel_for(
                range<1>{1},
                ext::oneapi::experimental::properties{
                    ext::oneapi::experimental::work_group_scratch_size(
                        sizeof(int))},
                [=](item<1> idx) {});
          });
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);
}

TEST_F(CommandGraphTest, MakeEdgeErrors) {
  // Set up some nodes in the graph
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

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
    experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);
    experimental::detail::node_impl &NodeAImpl = *getSyclObjImpl(NodeA);
    experimental::detail::node_impl &NodeBImpl = *getSyclObjImpl(NodeB);

    ASSERT_EQ(GraphImpl.MRoots.size(), 1lu);
    ASSERT_EQ(*GraphImpl.MRoots.begin(), &NodeAImpl);

    ASSERT_EQ(NodeAImpl.MSuccessors.size(), 1lu);
    ASSERT_EQ(NodeAImpl.MPredecessors.size(), 0lu);
    ASSERT_EQ(NodeAImpl.MSuccessors.front(), &NodeBImpl);

    ASSERT_EQ(NodeBImpl.MSuccessors.size(), 0lu);
    ASSERT_EQ(NodeBImpl.MPredecessors.size(), 1lu);
    ASSERT_EQ(NodeBImpl.MPredecessors.front(), &NodeAImpl);
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

TEST_F(CommandGraphTest, ProfilingException) {
  Graph.begin_recording(Queue);
  auto Event1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Event2 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
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

TEST_F(CommandGraphTest, ProfilingExceptionProperty) {
  Graph.begin_recording(Queue);
  auto Event1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  // Checks exception thrown if profiling is requested while profiling has
  // not been enabled during the graph building.
  auto GraphExecInOrder = Graph.finalize();
  queue QueueProfile{Dev, {sycl::property::queue::enable_profiling()}};
  auto EventInOrder = QueueProfile.submit(
      [&](handler &CGH) { CGH.ext_oneapi_graph(GraphExecInOrder); });
  QueueProfile.wait_and_throw();
  bool Success = true;
  try {
    EventInOrder
        .get_profiling_info<sycl::info::event_profiling::command_start>();
  } catch (sycl::exception &Exception) {
    ASSERT_FALSE(std::string(Exception.what())
                     .find("Profiling information is unavailable as the queue "
                           "associated with the event does not have the "
                           "'enable_profiling' property.") ==
                 std::string::npos);
    Success = false;
  }
  ASSERT_EQ(Success, false);
}

TEST_F(CommandGraphTest, ClusterLaunchException) {
  namespace syclex = sycl::ext::oneapi::experimental;

  syclex::properties cluster_launch_property{
      syclex::cuda::cluster_size<1>(sycl::range<1>{4})};

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.begin_recording(Queue);
    auto Event1 = Queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<TestKernel>(sycl::nd_range<1>({4096}, {32}),
                                   cluster_launch_property,
                                   [&](sycl::nd_item<1> it) {});
    });
    Queue.wait();
    Graph.end_recording(Queue);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  ASSERT_EQ(ExceptionCode, sycl::errc::invalid);
}

// Submits a command to a queue that has a dependency to a graph event
// associated with a different context.
TEST_F(CommandGraphTest, TransitiveRecordingWrongContext) {

  device Dev;
  context Ctx{Dev};
  context Ctx2{Dev};
  queue Q1{Ctx, Dev};
  queue Q2{Ctx2, Dev};

  ext::oneapi::experimental::command_graph Graph{Q1.get_context(),
                                                 Q1.get_device()};
  Graph.begin_recording(Q1);

  auto GraphEvent1 =
      Q1.submit([&](handler &CGH) { CGH.single_task<class Kernel1>([=] {}); });

  ASSERT_THROW(Q2.submit([&](handler &CGH) {
    CGH.depends_on(GraphEvent1);
    CGH.single_task<class Kernel2>([=] {});
  }),
               sycl::exception);
}

// Submits a command to a queue that has a dependency to a graph event
// associated with a different device.
TEST_F(CommandGraphTest, TransitiveRecordingWrongDevice) {

  auto devices = device::get_devices();

  // Test needs at least 2 devices available.
  if (devices.size() < 2) {
    GTEST_SKIP();
  }

  device &Dev1 = devices[0];
  device &Dev2 = devices[1];
  context Ctx{{Dev1, Dev2}};
  queue Q1{Ctx, Dev1};
  queue Q2{Ctx, Dev2};

  ext::oneapi::experimental::command_graph Graph{Q1.get_context(),
                                                 Q1.get_device()};
  Graph.begin_recording(Q1);

  auto GraphEvent1 =
      Q1.submit([&](handler &CGH) { CGH.single_task<class Kernel1>([=] {}); });

  ASSERT_THROW(Q2.submit([&](handler &CGH) {
    CGH.depends_on(GraphEvent1);
    CGH.single_task<class Kernel2>([=] {});
  }),
               sycl::exception);
}

// Submits a command to a queue that has a dependency to a different graph.
TEST_F(CommandGraphTest, RecordingWrongGraphDep) {
  device Dev;
  context Ctx{{Dev}};
  queue Q1{Ctx, Dev};
  queue Q2{Ctx, Dev};

  ext::oneapi::experimental::command_graph Graph1{Q1.get_context(),
                                                  Q1.get_device()};

  ext::oneapi::experimental::command_graph Graph2{Q1.get_context(),
                                                  Q1.get_device()};

  Graph1.begin_recording(Q1);
  Graph2.begin_recording(Q2);

  auto GraphEvent1 =
      Q1.submit([&](handler &CGH) { CGH.single_task<class Kernel1>([=] {}); });

  ASSERT_THROW(Q2.submit([&](handler &CGH) {
    CGH.depends_on(GraphEvent1);
    CGH.single_task<class Kernel2>([=] {});
  }),
               sycl::exception);
}

// Error when a dynamic command-group is used with a graph belonging to a
// different graph.
TEST_F(CommandGraphTest, DynamicCommandGroupWrongGraph) {
  experimental::command_graph Graph1{Queue.get_context(), Queue.get_device()};
  experimental::command_graph Graph2{Queue.get_context(), Queue.get_device()};
  auto CGF = [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); };

  experimental::dynamic_command_group DynCG(Graph2, {CGF});
  ASSERT_THROW(Graph1.add(DynCG), sycl::exception);
}

// Error when a non-kernel command-group is included in a dynamic command-group
TEST_F(CommandGraphTest, DynamicCommandGroupNotKernel) {
  int *Ptr = malloc_device<int>(1, Queue);
  auto CGF = [&](sycl::handler &CGH) { CGH.memset(Ptr, 1, 0); };

  experimental::command_graph Graph{Queue};
  ASSERT_THROW(experimental::dynamic_command_group DynCG(Graph, {CGF}),
               sycl::exception);
  sycl::free(Ptr, Queue);
}

// Error if edges are not the same for all command-groups in dynamic command
// group, test using graph limited events to create edges
TEST_F(CommandGraphTest, DynamicCommandGroupMismatchEventEdges) {
  size_t N = 32;
  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);

  experimental::command_graph Graph{Queue.get_context(), Queue.get_device()};

  Graph.begin_recording(Queue);

  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](item<1> Item) { PtrA[Item.get_id()] = 1; });
  });

  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](item<1> Item) { PtrB[Item.get_id()] = 4; });
  });

  Graph.end_recording();

  auto CGFA = [&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(N, [=](item<1> Item) { PtrA[Item.get_id()] += 2; });
  };

  auto CGFB = [&](handler &CGH) {
    CGH.depends_on(EventB);
    CGH.parallel_for(N, [=](item<1> Item) { PtrB[Item.get_id()] += 0xA; });
  };

  experimental::dynamic_command_group DynCG(Graph, {CGFA, CGFB});
  ASSERT_THROW(Graph.add(DynCG), sycl::exception);

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
}

// Test that an exception is thrown when a graph isn't created with buffer
// property, but buffers are used.
TEST_F(CommandGraphTest, DynamicCommandGroupBufferThrows) {
  size_t N = 32;
  std::vector<int> HostData(N, 0);
  buffer Buf{HostData};
  Buf.set_write_back(false);

  experimental::command_graph Graph{Queue.get_context(), Queue.get_device()};

  auto CGFA = [&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { Acc[Item.get_id()] = 2; });
  };

  auto CGFB = [&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { Acc[Item.get_id()] = 0xA; });
  };

  experimental::dynamic_command_group DynCG(Graph, {CGFA, CGFB});
  ASSERT_THROW(Graph.add(DynCG), sycl::exception);
}

// Test and exception is thrown when using a host-accessor to a buffer
// used in a non active CGF node in the graph.
TEST_F(CommandGraphTest, DynamicCommandGroupBufferHostAccThrows) {
  size_t N = 32;
  std::vector<int> HostData(N, 0);
  buffer Buf{HostData};
  Buf.set_write_back(false);

  int *Ptr = malloc_device<int>(N, Queue);

  {
    ext::oneapi::experimental::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {experimental::property::graph::assume_buffer_outlives_graph{}}};

    auto CGFA = [&](handler &CGH) {
      CGH.parallel_for(N, [=](item<1> Item) { Ptr[Item.get_id()] = 2; });
    };

    auto CGFB = [&](handler &CGH) {
      auto Acc = Buf.get_access<access::mode::write>(CGH);
      CGH.parallel_for(N, [=](item<1> Item) { Acc[Item.get_id()] = 0xA; });
    };

    experimental::dynamic_command_group DynCG(Graph, {CGFA, CGFB});
    ASSERT_NO_THROW(Graph.add(DynCG));

    ASSERT_THROW({ host_accessor HostAcc{Buf}; }, sycl::exception);
  }

  sycl::free(Ptr, Queue);
}

// Error if edges are not the same for all command-groups in dynamic command
// group, test using accessors to create edges
TEST_F(CommandGraphTest, DynamicCommandGroupMismatchAccessorEdges) {
  size_t N = 32;
  std::vector<int> HostData(N, 0);
  buffer BufA{HostData};
  buffer BufB{HostData};
  BufA.set_write_back(false);
  BufB.set_write_back(false);

  experimental::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {experimental::property::graph::assume_buffer_outlives_graph{}}};

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    auto AccA = BufA.get_access<access::mode::write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { AccA[Item.get_id()] = 1; });
  });

  Queue.submit([&](handler &CGH) {
    auto AccB = BufB.get_access<access::mode::write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { AccB[Item.get_id()] = 4; });
  });

  Graph.end_recording();

  auto CGFA = [&](handler &CGH) {
    auto AccA = BufA.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { AccA[Item.get_id()] += 2; });
  };

  auto CGFB = [&](handler &CGH) {
    auto AccB = BufB.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(N, [=](item<1> Item) { AccB[Item.get_id()] += 0xA; });
  };

  experimental::dynamic_command_group DynCG(Graph, {CGFA, CGFB});
  ASSERT_THROW(Graph.add(DynCG), sycl::exception);
}

// host and shared allocations are not currently supported by graphs, checks for
// correct exception behaviour.
TEST_F(CommandGraphTest, AsyncAllocKindExceptionCheck) {
  auto Context = Queue.get_context();
  auto Device = Queue.get_device();

  void *Ptr1 = nullptr;
  void *Ptr2 = nullptr;

  Graph.begin_recording(Queue);

  addAsyncAlloc<OperationPath::RecordReplay, usm::alloc::host>(Graph, Queue,
                                                               1024, Ptr1);
  addAsyncAlloc<OperationPath::RecordReplay, usm::alloc::shared>(Graph, Queue,
                                                                 1024, Ptr1);
  addAsyncAlloc<OperationPath::Shortcut, usm::alloc::host>(Graph, Queue, 1024,
                                                           Ptr2);
  addAsyncAlloc<OperationPath::Shortcut, usm::alloc::shared>(Graph, Queue, 1024,
                                                             Ptr2);

  Graph.end_recording();

  void *Ptr3 = nullptr;
  void *Ptr4 = nullptr;
  addAsyncAlloc<OperationPath::Explicit, usm::alloc::host>(Graph, Queue, 1024,
                                                           Ptr3);
  addAsyncAlloc<OperationPath::Explicit, usm::alloc::shared>(Graph, Queue, 1024,
                                                             Ptr4);
}
