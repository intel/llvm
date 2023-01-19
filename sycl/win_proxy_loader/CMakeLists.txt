
project(win_proxy_loader)
add_library(win_proxy_loader SHARED  win_proxy_loader.cpp)
if (WIN32)
  install(TARGETS win_proxy_loader
          RUNTIME DESTINATION "bin" COMPONENT win_proxy_loader
  )
endif()

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
endif()


# Handle the debug version for the Microsoft compiler as a special case by
# creating a debug version of the static library that uses the flags used by
# the SYCL runtime
if (MSVC)
  add_library(win_proxy_loaderd SHARED win_proxy_loader.cpp)
  target_compile_options(win_proxy_loaderd PRIVATE ${WINUNLOAD_CXX_FLAGS_DEBUG})
  target_compile_options(win_proxy_loader PRIVATE ${WINUNLOAD_CXX_FLAGS_RELEASE})
  target_link_libraries(win_proxy_loaderd PRIVATE shlwapi)
  target_link_libraries(win_proxy_loader PRIVATE shlwapi)
  if (WIN32)
     install(TARGETS win_proxy_loaderd
             RUNTIME DESTINATION "bin" COMPONENT win_proxy_loader
  )
  endif()
endif()


