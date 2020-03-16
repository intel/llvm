cmake_minimum_required(VERSION 2.8.9)
project (basic_collector)

file(GLOB SOURCES *.cpp)
include_directories(${XPTIFW_DIR}/include)
include_directories(${XPTI_DIR}/include)
include_directories(${XPTIFW_DIR}/samples/include)

remove_definitions(-DXPTI_STATIC_LIBRARY)
add_definitions(-DXPTI_API_EXPORTS)
add_library(basic_collector SHARED ${SOURCES})
add_dependencies(basic_collector xptifw)
if (MSVC)
  target_link_libraries(basic_collector
            PRIVATE xptifw
            )
else()
  target_link_libraries(basic_collector
            PRIVATE xptifw
            PRIVATE dl
            )
endif()
# Set the location of the library installation
install(TARGETS basic_collector DESTINATION ${CMAKE_BINARY_DIR})
