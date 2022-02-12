# PI Level Zero plugin library

if(MSVC)
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LOADER
      "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
  message(STATUS "Download Level Zero loader and headers from github.com")
  if (CMAKE_C_COMPILER)
    list(APPEND AUX_CMAKE_FLAGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  if (CMAKE_CXX_COMPILER)
    list(APPEND AUX_CMAKE_FLAGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_build)
  set(LEVEL_ZERO_LOADER_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/level_zero/level_zero_loader")
  if (NOT DEFINED SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE)
    set(SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE ${SYCL_EXTERNAL_PROJECTS_SKIP_AUTO_UPDATE})
  endif()
  if (WIN32)
    set(LEVEL_ZERO_LOADER_SOURCE_VERSION v1.2.3)
  else()
    set(LEVEL_ZERO_LOADER_SOURCE_VERSION v1.4.1)
  endif()

  #TODO: Replace ExternalProject with FetchContent for better maintainance and
  #      cmake files simplification
  include(ExternalProject)
  ExternalProject_Add(level-zero-loader
    GIT_REPOSITORY    https://github.com/oneapi-src/level-zero.git
    GIT_TAG           ${LEVEL_ZERO_LOADER_SOURCE_VERSION}
    UPDATE_DISCONNECTED ${SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE}
    SOURCE_DIR        ${LEVEL_ZERO_LOADER_SOURCE_DIR}
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_build"
    INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_install"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_INSTALL_LIBDIR:PATH=lib${LLVM_LIBDIR_SUFFIX}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               ${AUX_CMAKE_FLAGS}
               LOG_DOWNLOAD 1
               LOG_UPDATE 1
               LOG_CONFIGURE 1
               LOG_BUILD 1
               LOG_INSTALL 1
    STEP_TARGETS      configure,build,install
    DEPENDS           OpenCL-Headers
    BUILD_BYPRODUCTS ${LEVEL_ZERO_LOADER}
  )
  ExternalProject_Add_Step(level-zero-loader llvminstall
    COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/lib/ ${LLVM_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/include/ ${LLVM_BINARY_DIR}/include/sycl
    COMMENT "Installing level-zero-loader into the LLVM binary directory"
    DEPENDEES install
  )

  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_install/"
    DESTINATION "."
    COMPONENT level-zero-loader
  )

  set(LEVEL_ZERO_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader_install/include/
	  CACHE INTERNAL "Path containing Level_Zero header files.")
else()
  file(GLOB LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}*")
  get_filename_component(LEVEL_ZERO_LIB_NAME ${LEVEL_ZERO_LIBRARY} NAME)
  add_custom_target(level-zero-loader
      DEPENDS
	  ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LEVEL_ZERO_LIB_NAME}
	  ${LLVM_BINARY_DIR}/include/sycl/level_zero
  )
  add_custom_command(
      OUTPUT
	  ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LEVEL_ZERO_LIB_NAME}
	  ${LLVM_BINARY_DIR}/include/sycl/level_zero
      COMMENT
	  "Copying Level Zero loader and headers"
      COMMAND
	  ${CMAKE_COMMAND} -E copy ${LEVEL_ZERO_LIBRARY_SRC} ${LLVM_LIBRARY_OUTPUT_INTDIR}
      COMMAND
	  ${CMAKE_COMMAND} -E copy_directory ${LEVEL_ZERO_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include/sycl
      DEPENDS
	  ${LEVEL_ZERO_LIBRARY}
          ${LEVEL_ZERO_INCLUDE_DIR}
  )
endif()

list(APPEND SYCL_TOOLCHAIN_DEPLOY_COMPONENTS level-zero-loader level-zero-headers)

include_directories("${LEVEL_ZERO_INCLUDE_DIR}")

add_library (LevelZeroLoader-Headers INTERFACE)
add_library (LevelZeroLoader::Headers ALIAS LevelZeroLoader-Headers)
target_include_directories(LevelZeroLoader-Headers
  INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)


include_directories("${sycl_inc_dir}")

add_library(pi_level_zero SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/pi_level_zero.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/pi_level_zero.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/usm_allocator.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/usm_allocator.hpp"
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_level_zero PRIVATE __SYCL_BUILD_SYCL_DLL)
  # Install stripped PDB
  add_stripped_pdb(pi_level_zero)
else()
  # we set the visibility of all symbols 'hidden' by default.
  # In pi.h file, we set exported symbols with visibility==default individually
  target_compile_options(pi_level_zero PUBLIC -fvisibility=hidden)

  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

  # Filter symbols based on the scope defined in the script file,
  # and export pi* function symbols in the library.
  target_link_libraries( pi_level_zero
    PRIVATE "-Wl,--version-script=${linker_script}"
  )
endif()

if (TARGET level-zero-loader)
  add_dependencies(pi_level_zero level-zero-loader)
endif()

add_dependencies(sycl-toolchain pi_level_zero)
target_link_libraries(pi_level_zero
  PRIVATE
    "${LEVEL_ZERO_LOADER}"
    OpenCL-Headers
)

if (UNIX)
  target_link_libraries(pi_level_zero PRIVATE pthread)
endif()

add_common_options(pi_level_zero)

install(TARGETS pi_level_zero
  LIBRARY DESTINATION "lib" COMPONENT pi_level_zero
  RUNTIME DESTINATION "bin" COMPONENT pi_level_zero)
