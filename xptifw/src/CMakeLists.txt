# Fetch third party headers
include(FetchContent)

# The sycl target should download or find emhash but may not
# for standalone builds.
if (NOT TARGET emhash::emhash)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../sycl/cmake/modules")
  include(FetchEmhash)
endif()

set(PHMAP_REQUIRED_VER "1.3.12")

find_path(PHMAP_LOC phmap_base.h PATH_SUFFIXES parallel_hashmap)
if(PHMAP_LOC)
 file(READ "${PHMAP_LOC}/phmap_config.h" PHMAP_CONFIG_H)
 string(REGEX MATCH "#define PHMAP_VERSION_MAJOR [0-9]+" PHMAP_VER_MAJOR_LINE "${PHMAP_CONFIG_H}")
 string(REGEX MATCH "#define PHMAP_VERSION_MINOR [0-9]+" PHMAP_VER_MINOR_LINE "${PHMAP_CONFIG_H}")
 string(REGEX MATCH "#define PHMAP_VERSION_PATCH [0-9]+" PHMAP_VER_PATCH_LINE "${PHMAP_CONFIG_H}")

 # We mached the whole line, extract out the number.
 string(REGEX REPLACE ".*PHMAP_VERSION_MAJOR ([0-9]+).*" "\\1" PHMAP_VER_MAJOR "${PHMAP_VER_MAJOR_LINE}")
 string(REGEX REPLACE ".*PHMAP_VERSION_MINOR ([0-9]+).*" "\\1" PHMAP_VER_MINOR "${PHMAP_VER_MINOR_LINE}")
 string(REGEX REPLACE ".*PHMAP_VERSION_PATCH ([0-9]+).*" "\\1" PHMAP_VER_PATCH "${PHMAP_VER_PATCH_LINE}")
 set(PHMAP_VER "${PHMAP_VER_MAJOR}.${PHMAP_VER_MINOR}.${PHMAP_VER_PATCH}")
 if(${PHMAP_VER} VERSION_LESS ${PHMAP_REQUIRED_VER})
   message(WARNING "Found system install of parallel-hashmap, but it is version ${PHMAP_VER}"
     " and ${PHMAP_REQUIRED_VER} or later is required, building from source")
   set(PHMAP_LOC 0)
 endif()
endif()
if(PHMAP_LOC)
  set(XPTIFW_PARALLEL_HASHMAP_HEADERS "${PHMAP_LOC}")
  message(STATUS "Using system parallel-hashmap version ${PHMAP_VER}")
else() 
  set(PARALLEL_HASHMAP_REPO https://github.com/greg7mdp/parallel-hashmap.git)
  message(STATUS "Will fetch parallel-hashmap version ${PHMAP_REQUIRED_VER} from ${PARALLEL_HASHMAP_REPO}")
  # When updating the tag, please check phmap_config.h and update PHMAP_REQUIRED_VER above as well.
  FetchContent_Declare(parallel-hashmap
    GIT_REPOSITORY    ${PARALLEL_HASHMAP_REPO}
    GIT_TAG           8a889d3699b3c09ade435641fb034427f3fd12b6
  )
  
  FetchContent_GetProperties(parallel-hashmap)
  FetchContent_MakeAvailable(parallel-hashmap)
  set(XPTIFW_PARALLEL_HASHMAP_HEADERS "${parallel-hashmap_SOURCE_DIR}")
endif()


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
    $<BUILD_INTERFACE:${XPTIFW_PARALLEL_HASHMAP_HEADERS}>
    $<BUILD_INTERFACE:${XPTI_DIR}/include>
  )

  target_link_libraries(${LIB_NAME} PUBLIC emhash::emhash)

  find_package(Threads REQUIRED)
  target_link_libraries(${LIB_NAME} PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)

  if (XPTI_ENABLE_TBB)
    add_dependencies(${LIB_NAME} tbb)
    target_compile_definitions(${LIB_NAME} PRIVATE XPTI_USE_TBB)
    target_link_libraries(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:tbb>)
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
