set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(ProgramManagerTests OBJECT
  EliminatedArgMask.cpp
  itt_annotations.cpp
)

