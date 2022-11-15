cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)
project(unified_runtime)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#Define a path for custom commands to work around MSVC
set(CUSTOM_COMMAND_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
if(MSVC)
    #MSVC implicitly adds $<CONFIG> to the output path
    set(CUSTOM_COMMAND_BINARY_DIR ${CUSTOM_COMMAND_BINARY_DIR}/$<CONFIG>)
endif()

#CXX compiler support
if(NOT MSVC)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support.  Please use a different C++ compiler.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fPIC")
endif()

#MSVC compile flags
if(MSVC)
    string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    # treat warnings as errors
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX /W3 /wd4996")

    # enable multi-process compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# Allow custom third_party folder
if(NOT DEFINED THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

# Option to disable tests
option(${PROJECT_NAME}_BUILD_TESTS "Build unit tests." ON)

add_subdirectory(${THIRD_PARTY_DIR})

# Copy third_party_binaries to output BIN folder
#add_custom_target(copy_third_party_files)

#add_custom_command(
#    TARGET copy_third_party_files
#    POST_BUILD
#)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(source)
