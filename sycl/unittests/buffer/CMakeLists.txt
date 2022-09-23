add_definitions(-gdwarf-4 -O0)
add_sycl_unittest(BufferTests OBJECT
  BufferLocation.cpp
  Image.cpp
  BufferDestructionCheck.cpp
)
