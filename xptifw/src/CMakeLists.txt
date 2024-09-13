# Fetch third party headers
set(EMHASH_REPO https://github.com/ktprime/emhash)
message(STATUS "Will fetch emhash from ${EMHASH_REPO}")
FetchContent_Declare(emhash-headers
  GIT_REPOSITORY    ${EMHASH_REPO}
  GIT_TAG           96dcae6fac2f5f90ce97c9efee61a1d702ddd634
)

FetchContent_GetProperties(emhash-headers)
if(NOT emhash-headers_POPULATED)
  FetchContent_Populate(emhash-headers)
endif()

set(XPTIFW_EMHASH_HEADERS "${emhash-headers_SOURCE_DIR}")

set(PARALLEL_HASHMAP_REPO https://github.com/greg7mdp/parallel-hashmap.git)
message(STATUS "Will fetch parallel-hashmap from ${PARALLEL_HASHMAP_REPO}")
FetchContent_Declare(parallel-hashmap
  GIT_REPOSITORY    ${PARALLEL_HASHMAP_REPO}
  GIT_TAG           8a889d3699b3c09ade435641fb034427f3fd12b6
)

FetchContent_GetProperties(parallel-hashmap)
FetchContent_MakeAvailable(parallel-hashmap)

set(XPTIFW_PARALLEL_HASHMAP_HEADERS "${parallel-hashmap_SOURCE_DIR}")

file(GLOB SOURCES *.cpp)

function(add_xpti_library LIB_NAME)
  remove_definitions(-DXPTI_STATIC_LIBRARY)
  add_library(${LIB_NAME} SHARED ${SOURCES})

  if (${LIB_NAME} MATCHES "xptifwd")
    set_property(TARGET ${LIB_NAME} PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
  else()
    set_property(TARGET ${LIB_NAME} PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  endif()

  target_compile_definitions(${LIB_NAME} PRIVATE -DXPTI_API_EXPORTS)
  target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${XPTIFW_DIR}/include>
    $<BUILD_INTERFACE:${XPTIFW_EMHASH_HEADERS}>
    $<BUILD_INTERFACE:${XPTIFW_PARALLEL_HASHMAP_HEADERS}>
    $<BUILD_INTERFACE:${XPTI_DIR}/include>
  )

  find_package(Threads REQUIRED)
  target_link_libraries(${LIB_NAME} PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)

  if (XPTI_ENABLE_TBB)
    add_dependencies(${LIB_NAME} tbb)
    target_compile_definitions(${LIB_NAME} PRIVATE XPTI_USE_TBB)
    target_link_libraries(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:tbb>)
  endif()

  if (SYCL_USE_LIBCXX)
      target_link_libraries(${LIB_NAME} PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -Wl,-rpath,${SYCL_LIBCXX_LIBRARY_PATH} -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
  endif()

  # Set the location of the library installation
  include(GNUInstallDirs)
  install(TARGETS ${LIB_NAME}
    RUNTIME DESTINATION bin COMPONENT xptifw
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xptifw
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xptifw
  )
endfunction(add_xpti_library)

# MSVC provides two incompatible build variants for its CRT: release and debug
# To avoid potential issues in user code we also need to provide two kinds
# of XPTIFW for release and debug configurations.
add_xpti_library(xptifw)
if (MSVC)
  add_xpti_library(xptifwd)
endif()
