set(CMAKE_CXX_EXTENSIONS OFF)
add_sycl_unittest(ProgramManagerArgMaskTests OBJECT
  EliminatedArgMask.cpp
)

