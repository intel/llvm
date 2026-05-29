# DPCPP Version number information

if(NOT DEFINED DPCPP_VERSION_MAJOR)
  set(DPCPP_VERSION_MAJOR 7)
endif()

# Install directory for all DPC++ compiler tools.  On Linux, tools are placed
# under a versioned subdirectory to avoid conflicts with a system LLVM
# installation.  On other platforms the standard bin/ directory is used.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(DPCPP_INSTALL_INTERNAL_BINDIR "lib/dpcpp-${DPCPP_VERSION_MAJOR}/bin")
else()
  set(DPCPP_INSTALL_INTERNAL_BINDIR "bin")
endif()
if(NOT DEFINED DPCPP_VERSION_MINOR)
  set(DPCPP_VERSION_MINOR 1)
endif()
if(NOT DEFINED DPCPP_VERSION_PATCH)
  set(DPCPP_VERSION_PATCH 0)
endif()
if(NOT DEFINED PRE_RELEASE)
  set(PRE_RELEASE 1)
endif()
