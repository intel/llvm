set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(ReusableEventsTests OBJECT
  ReusableEventsFixture.cpp
  ReusableEventsBinding.cpp
  ReusableEventsLifetime.cpp
  ReusableEventsQueue.cpp
  ReusableEventsScheduler.cpp
  ReusableEventsIpc.cpp
  ReusableEventsGraph.cpp
)
