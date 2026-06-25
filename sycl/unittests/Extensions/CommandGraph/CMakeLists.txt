set(CMAKE_CXX_EXTENSIONS OFF)
if(WIN32)
  # https://github.com/intel/llvm/issues/22425
  return()
endif()
add_sycl_unittest(CommandGraphExtensionTests OBJECT
  Barrier.cpp
  CommandGraph.cpp
  CommonReferenceSemantics.cpp
  Exceptions.cpp
  InOrderQueue.cpp
  MultiThreaded.cpp
  Queries.cpp
  Regressions.cpp
  Subgraph.cpp
  TopologicalSort.cpp
  Update.cpp
  Properties.cpp
  LinearGraphOptimization.cpp
)
