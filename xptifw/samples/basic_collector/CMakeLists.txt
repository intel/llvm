file(GLOB SOURCES *.cpp)
include_directories(${XPTIFW_DIR}/include)
include_directories(${XPTI_DIR}/include)
include_directories(${XPTIFW_DIR}/samples/include)

remove_definitions(-DXPTI_STATIC_LIBRARY)
add_definitions(-DXPTI_API_EXPORTS)
add_library(basic_collector SHARED ${SOURCES})
add_dependencies(basic_collector xptifw)
target_link_libraries(basic_collector PRIVATE xptifw)
if(UNIX)
  target_link_libraries(basic_collector PRIVATE dl)
endif()

if (XPTI_ENABLE_TBB)
  target_link_libraries(basic_collector PRIVATE tbb)
endif()
# Set the location of the library installation
install(TARGETS basic_collector DESTINATION ${CMAKE_BINARY_DIR})
