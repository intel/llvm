project(pi_win_proxy_loader)
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc
               PROPERTY COMPILE_DEFINITIONS
	       "RC_VERSION_FIELD_1=${SYCL_MAJOR_VERSION}"
               "RC_VERSION_FIELD_2=${SYCL_MAJOR_VERSION}"
               "RC_VERSION_FIELD_3=${SYCL_PATCH_VERSION}"
               "RC_VERSION_FIELD_4=${SYCL_DEV_ABI_VERSION}"
               "RC_FILE_VERSION=\"${SYCL_VERSION_STRING}\""
               "RC_INTERNAL_NAME=\"pi_win_proxy_loader\""
               "RC_PRODUCT_NAME=\"pi_win_proxy_loader\""
               "RC_PRODUCT_VERSION=\"${SYCL_VERSION_STRING}\""
               "RC_COPYRIGHT=\"Copyright (C) 2023 Intel Inc. All Rights Reserved\"")
configure_file(../../llvm/resources/windows_version_resource.rc ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc @ONLY)
add_library(pi_win_proxy_loader SHARED  pi_win_proxy_loader.cpp  ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc)
install(TARGETS pi_win_proxy_loader
        RUNTIME DESTINATION "bin" COMPONENT pi_win_proxy_loader
)

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  set(WINUNLOAD_CXX_FLAGS "")
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(WINUNLOAD_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MDd" "" WINUNLOAD_CXX_FLAGS "${WINUNLOAD_CXX_FLAGS}")
    string(REPLACE "/MTd" "" WINUNLOAD_CXX_FLAGS "${WINUNLOAD_CXX_FLAGS}")
  else()
    if (CMAKE_BUILD_TYPE MATCHES "Release")
      set(WINUNLOAD_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    elseif (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
      set(WINUNLOAD_CXX_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    elseif (CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
      set(WINUNLOAD_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    endif()
    string(REPLACE "/MD" "" WINUNLOAD_CXX_FLAGS "${WINUNLOAD_CXX_FLAGS}")
    string(REPLACE "/MT" "" WINUNLOAD_CXX_FLAGS "${WINUNLOAD_CXX_FLAGS}")
  endif()

  # target_compile_options requires list of options, not a string
  string(REPLACE " " ";" WINUNLOAD_CXX_FLAGS "${WINUNLOAD_CXX_FLAGS}")

  set(WINUNLOAD_CXX_FLAGS_RELEASE "${WINUNLOAD_CXX_FLAGS};/MD")
  set(WINUNLOAD_CXX_FLAGS_DEBUG "${WINUNLOAD_CXX_FLAGS};/MDd")

  # CMake automatically applies these flags to all targets. To override this
  # behavior, options lists are reset.
  set(CMAKE_CXX_FLAGS_RELEASE "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
  set(CMAKE_CXX_FLAGS_DEBUG "")

# Handle the debug version for the Microsoft compiler as a special case by
# creating a debug version of the static library that uses the flags used by
# the SYCL runtime
  add_library(pi_win_proxy_loaderd SHARED pi_win_proxy_loader.cpp ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc)
  target_compile_options(pi_win_proxy_loaderd PRIVATE ${WINUNLOAD_CXX_FLAGS_DEBUG})
  target_compile_options(pi_win_proxy_loader PRIVATE ${WINUNLOAD_CXX_FLAGS_RELEASE})
  target_link_libraries(pi_win_proxy_loaderd PRIVATE shlwapi)
  target_link_libraries(pi_win_proxy_loader PRIVATE shlwapi)
  # 0x2000: LOAD_LIBRARY_SAFE_CURRENT_DIRS flag. Using this flag means that loading dependency DLLs
  # from the current directory is only allowed if it is under a directory in the Safe load list.
  target_link_options(pi_win_proxy_loaderd PRIVATE /DEPENDENTLOADFLAG:0x2000)
  target_link_options(pi_win_proxy_loader PRIVATE /DEPENDENTLOADFLAG:0x2000)
  install(TARGETS pi_win_proxy_loaderd
          RUNTIME DESTINATION "bin" COMPONENT pi_win_proxy_loader)
endif()


