cmake_minimum_required(VERSION 2.8.9)
project (xptifw)

file(GLOB SOURCES *.cpp)
include_directories(${XPTIFW_DIR}/include)
include_directories(${XPTI_DIR}/include)

remove_definitions(-DXPTI_STATIC_LIBRARY)
add_definitions(-DXPTI_API_EXPORTS)
add_library(xptifw SHARED ${SOURCES})
add_dependencies(xptifw tbb)
if (MSVC)
  target_link_libraries(xptifw 
            PRIVATE tbb
            )
else()
  target_link_libraries(xptifw 
            PRIVATE tbb
            PRIVATE dl
            )
endif()
# Set the location of the library installation
install(TARGETS xptifw DESTINATION ${CMAKE_BINARY_DIR})
