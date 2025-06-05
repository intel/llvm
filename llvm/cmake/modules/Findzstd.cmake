# Try to find the zstd library
#
# If successful, the following variables will be defined:
# zstd_INCLUDE_DIR
# zstd_LIBRARY
# zstd_STATIC_LIBRARY
# zstd_FOUND
#
# Additionally, one of the following import targets will be defined:
# zstd::libzstd_shared
# zstd::libzstd_static

# Function to get zstd version.
function(get_zstd_version_string zstd_INCLUDE_DIR OUT_VAR)

    # Check if zstd.h file is present. Expectation is that this function will only
    # be called if zstd is found and the include directory is valid.
    if(NOT EXISTS "${zstd_INCLUDE_DIR}/zstd.h")
      message(FATAL_ERROR "zstd.h not found in ${zstd_INCLUDE_DIR}. "
        "Please set zstd_INCLUDE_DIR to the directory containing zstd.h.")
    endif()

    # Read the version string from zstd.h.
    # The version is defined as macros ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR,
    # and ZSTD_VERSION_RELEASE in zstd.h.
    file(READ "${zstd_INCLUDE_DIR}/zstd.h" content)
    string(REGEX MATCH "#define[ \t]+ZSTD_VERSION_MAJOR[ \t]+([0-9]+)" _major_match "${content}")
    string(REGEX MATCH "#define[ \t]+ZSTD_VERSION_MINOR[ \t]+([0-9]+)" _minor_match "${content}")
    string(REGEX MATCH "#define[ \t]+ZSTD_VERSION_RELEASE[ \t]+([0-9]+)" _patch_match "${content}")

    if(_major_match AND _minor_match AND _patch_match)
        string(REGEX REPLACE ".*#define[ \t]+ZSTD_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _major "${_major_match}")
        string(REGEX REPLACE ".*#define[ \t]+ZSTD_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _minor "${_minor_match}")
        string(REGEX REPLACE ".*#define[ \t]+ZSTD_VERSION_RELEASE[ \t]+([0-9]+).*" "\\1" _patch "${_patch_match}")
        set(_version "${_major}.${_minor}.${_patch}")
        set(${OUT_VAR} "${_version}" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

if(MSVC OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
  set(zstd_STATIC_LIBRARY_SUFFIX "_static\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
else()
  set(zstd_STATIC_LIBRARY_SUFFIX "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
endif()

find_path(zstd_INCLUDE_DIR NAMES zstd.h)
find_library(zstd_LIBRARY NAMES zstd zstd_static)
find_library(zstd_STATIC_LIBRARY NAMES
  zstd_static
  "${CMAKE_STATIC_LIBRARY_PREFIX}zstd${CMAKE_STATIC_LIBRARY_SUFFIX}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    zstd DEFAULT_MSG
    zstd_LIBRARY zstd_INCLUDE_DIR
)

# Get zstd version.
if (zstd_FOUND AND zstd_INCLUDE_DIR)
  get_zstd_version_string("${zstd_INCLUDE_DIR}" zstd_VERSION_STRING)
  message(STATUS "Found zstd version ${zstd_VERSION_STRING}.")
endif()

if(zstd_FOUND)
  if(zstd_LIBRARY MATCHES "${zstd_STATIC_LIBRARY_SUFFIX}$")
    set(zstd_STATIC_LIBRARY "${zstd_LIBRARY}")
  elseif (NOT TARGET zstd::libzstd_shared)
    add_library(zstd::libzstd_shared SHARED IMPORTED)
    if(MSVC OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
      include(GNUInstallDirs) # For CMAKE_INSTALL_LIBDIR and friends.
      # IMPORTED_LOCATION is the path to the DLL and IMPORTED_IMPLIB is the "library".
      get_filename_component(zstd_DIRNAME "${zstd_LIBRARY}" DIRECTORY)
      if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "" AND NOT "${CMAKE_INSTALL_BINDIR}" STREQUAL "")
        string(REGEX REPLACE "${CMAKE_INSTALL_LIBDIR}$" "${CMAKE_INSTALL_BINDIR}" zstd_DIRNAME "${zstd_DIRNAME}")
      endif()
      get_filename_component(zstd_BASENAME "${zstd_LIBRARY}" NAME)
      string(REGEX REPLACE "\\${CMAKE_LINK_LIBRARY_SUFFIX}$" "${CMAKE_SHARED_LIBRARY_SUFFIX}" zstd_BASENAME "${zstd_BASENAME}")
      set_target_properties(zstd::libzstd_shared PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
          IMPORTED_LOCATION "${zstd_DIRNAME}/${zstd_BASENAME}"
          IMPORTED_IMPLIB "${zstd_LIBRARY}")
      unset(zstd_DIRNAME)
      unset(zstd_BASENAME)
    else()
      set_target_properties(zstd::libzstd_shared PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
          IMPORTED_LOCATION "${zstd_LIBRARY}")
    endif()
  endif()
  if(zstd_STATIC_LIBRARY MATCHES "${zstd_STATIC_LIBRARY_SUFFIX}$" AND
     NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static STATIC IMPORTED)
    set_target_properties(zstd::libzstd_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
        IMPORTED_LOCATION "${zstd_STATIC_LIBRARY}")
  endif()
endif()

unset(zstd_STATIC_LIBRARY_SUFFIX)

mark_as_advanced(zstd_INCLUDE_DIR zstd_LIBRARY zstd_STATIC_LIBRARY)
