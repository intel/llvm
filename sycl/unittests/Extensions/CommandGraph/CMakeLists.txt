set(CMAKE_CXX_EXTENSIONS OFF)

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
)
