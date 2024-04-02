set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(CommandGraphExtensionTests OBJECT
  Barrier.cpp
  CommandGraph.cpp
  Exceptions.cpp
  InOrderQueue.cpp
  MultiThreaded.cpp
  Queries.cpp
  Regressions.cpp
  Subgraph.cpp
  Update.cpp
)
