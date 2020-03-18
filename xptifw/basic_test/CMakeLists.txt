cmake_minimum_required(VERSION 2.8.9)
project (run_test)

file(GLOB SOURCES *.cpp *.hpp)
include_directories(${XPTIFW_DIR}/include)
include_directories(${XPTI_DIR}/include)

remove_definitions(-DXPTI_STATIC_LIBRARY)
add_definitions(-DXPTI_API_EXPORTS -g -O3)
add_executable(run_test ${SOURCES})
add_dependencies(run_test xptifw)
target_link_libraries(run_test PRIVATE xptifw)
if(UNIX)
  target_link_libraries(run_test PRIVATE dl)
endif()

if (XPTI_ENABLE_TBB)
  target_link_libraries(run_test PRIVATE tbb)
endif()

# Set the location of the library installation
install(TARGETS run_test DESTINATION ${CMAKE_BINARY_DIR})
