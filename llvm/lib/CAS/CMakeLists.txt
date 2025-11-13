
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # We append -fno-lifetime-dse in HandleLLVMOptions.cmake
  # append("-fno-lifetime-dse" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  # But it is causing link failure with llvm::StdThreadPool::asyncEnqueue
  string(REPLACE "-fno-lifetime-dse" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  DatabaseFile.cpp
  InMemoryCAS.cpp
  MappedFileRegionArena.cpp
  ObjectStore.cpp
  OnDiskCommon.cpp
  OnDiskDataAllocator.cpp
  OnDiskGraphDB.cpp
  OnDiskKeyValueDB.cpp
  OnDiskTrieRawHashMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  Support
)
