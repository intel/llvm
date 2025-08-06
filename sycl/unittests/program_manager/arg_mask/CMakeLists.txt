set(CMAKE_CXX_EXTENSIONS OFF)
add_sycl_unittests(ProgramManagerArgMaskTests OBJECT
  EliminatedArgMask.cpp
)
