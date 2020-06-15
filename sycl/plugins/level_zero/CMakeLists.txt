# PI Level0 plugin library

if (NOT DEFINED L0_LIBRARY OR NOT DEFINED L0_INCLUDE_DIR)
  message(STATUS "Download Level Zero loader and headers from github.com")
  if(MSVC)
    set(L0_LIBRARY
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
  else()
    set(L0_LIBRARY
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  if (CMAKE_C_COMPILER)
      list(APPEND AUX_CMAKE_FLAGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  if (CMAKE_CXX_COMPILER)
      list(APPEND AUX_CMAKE_FLAGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/l0_loader_build)
  ExternalProject_Add(l0-loader
    GIT_REPOSITORY    https://github.com/oneapi-src/level-zero.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Level0/l0_loader"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/l0_loader_build"
    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}/l0_loader_install"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIRS}
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_INSTALL_LIBDIR:PATH=lib${LLVM_LIBDIR_SUFFIX}
               ${AUX_CMAKE_FLAGS}
    STEP_TARGETS      configure,build,install
    DEPENDS           ocl-headers
    BUILD_BYPRODUCTS ${L0_LIBRARY}
  )
  ExternalProject_Add_Step(l0-loader llvminstall
    COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}
    COMMENT "Installing l0-loader into the LLVM binary directory"
    DEPENDEES install
  )

  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/l0_loader_install/"
    DESTINATION "."
    COMPONENT l0-loader
  )

  list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS l0-loader)
else()
  include_directories("${L0_INCLUDE_DIR}")
  add_custom_target(l0-loader
    DEPENDS ${L0_LIBRARY} COMMENT "Using provided L0 Loader")
endif()

add_library (L0Loader-Headers INTERFACE)
add_library (L0Loader::Headers ALIAS L0Loader-Headers)
target_include_directories(L0Loader-Headers
  INTERFACE "${L0_INCLUDE_DIR}"
)

include_directories("${sycl_inc_dir}")
include_directories(${OPENCL_INCLUDE})

add_library(pi_level0 SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/pi_level0.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/pi_level0.hpp"
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_level0 PRIVATE __SYCL_BUILD_SYCL_DLL)
else()
  # we set the visibility of all symbols 'hidden' by default.
  # In pi.h file, we set exported symbols with visibility==default individually
  target_compile_options(pi_level0 PUBLIC -fvisibility=hidden)

  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

  # Filter symbols based on the scope defined in the script file,
  # and export pi* function symbols in the library.
  target_link_libraries( pi_level0
    PRIVATE "-Wl,--version-script=${linker_script}"
  )
endif()

if (TARGET l0-loader)
  add_dependencies(pi_level0 l0-loader)
endif()
  add_dependencies(sycl-toolchain pi_level0)

target_link_libraries(pi_level0 PRIVATE "${L0_LIBRARY}")
if (UNIX)
  target_link_libraries(pi_level0 PRIVATE pthread)
endif()

add_common_options(pi_level0)

install(TARGETS pi_level0
    LIBRARY DESTINATION "lib" COMPONENT pi_level0
    RUNTIME DESTINATION "bin" COMPONENT pi_level0)
