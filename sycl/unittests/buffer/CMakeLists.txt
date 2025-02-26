add_sycl_unittest(BufferTests OBJECT
  PARTIAL_SOURCES_INTENDED
  BufferLocation.cpp
  Image.cpp
  BufferReleaseBase.cpp
  MemChannel.cpp
  KernelArgMemObj.cpp
  SubbufferLargeSize.cpp
  Properties.cpp
)
