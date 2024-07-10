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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph2.end_recording();

  auto Event = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

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

TEST_F(CommandGraphTest, ProfilingExceptionProperty) {
  Graph.begin_recording(Queue);
  auto Event1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
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
      cgh.parallel_for<TestKernel<>>(sycl::nd_range<1>({4096}, {32}),
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
