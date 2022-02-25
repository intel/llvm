set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(ProgramManagerTests OBJECT
  BuildLog.cpp
  EliminatedArgMask.cpp
  itt_annotations.cpp
  SubDevices.cpp
)

