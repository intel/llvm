include(ExternalProject)

if(DEFINED RSBench_BIN)
  message("-- Use RSBench binaries from ${RSBench_BIN}")
  add_custom_target(RSBench ALL
    DEPENDS   ${RSBench_BIN}
    COMMAND   ${CMAKE_COMMAND} -E copy_directory
                 ${RSBench_BIN}
                 ${CMAKE_CURRENT_BINARY_DIR}/Output
    COMMENT   "Deploying prebuilt RSBench binaries..."
  )
elseif(DEFINED RSBench_SRC)
  message("-- Use RSBench built from sources in ${RSBench_SRC}")
  ExternalProject_Add(RSBench
    SOURCE_DIR        ${RSBench_SRC}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     make CC=${CMAKE_CXX_COMPILER}
                      CXXFLAGS=${CMAKE_CXX_FLAGS}
                      -C ${RSBench_SRC}/sycl
    BUILD_IN_SOURCE   ON
    INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory
                        ${RSBench_SRC}/sycl
                        ${CMAKE_CURRENT_BINARY_DIR}/Output
    STEP_TARGETS      build, install
    COMMENT           "Building RSBench sources from ${RSBench_SRC}..."
    LOG_BUILD 1
    LOG_INSTALL 1
  )
else(DEFINED RSBench_BIN AND DEFINED RSBench_SRC)

  if(NOT DEFINED RSBench_URL)
    set(RSBench_URL https://github.com/ANL-CESAR/RSBench.git)
  endif(NOT DEFINED RSBench_URL)

  if(NOT DEFINED RSBench_TAG)
    set(RSBench_TAG origin/master)
  endif(NOT DEFINED RSBench_TAG)

  message("-- Use RSBench built from sources under ${RSBench_URL} (${RSBench_TAG})")
  set(RSBench_SRC ${CMAKE_CURRENT_BINARY_DIR}/src)
  ExternalProject_Add(RSBench
    GIT_REPOSITORY    ${RSBench_URL}
    GIT_TAG           ${RSBench_TAG}
    SOURCE_DIR        ${RSBench_SRC}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     make CC=${CMAKE_CXX_COMPILER}
                      CXXFLAGS=${CMAKE_CXX_FLAGS}
                      -C ${RSBench_SRC}/sycl
    BUILD_IN_SOURCE   ON
    INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory
                        ${RSBench_SRC}/sycl
                        ${CMAKE_CURRENT_BINARY_DIR}/Output
    STEP_TARGETS      build, install
    COMMENT           "Downloading from ${RSBench_URL} (${RSBench_TAG}) and build RSBench..."
    LOG_DOWNLOAD 1
    LOG_UPDATE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
endif()

set_property(GLOBAL APPEND PROPERTY TEST_SUITE_TARGETS RSBench)
