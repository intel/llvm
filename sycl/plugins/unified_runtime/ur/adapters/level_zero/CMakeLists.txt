set(TARGET_NAME ur_level_zero)

add_library(${TARGET_NAME}
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_level_zero.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_level_zero.hpp
        # These are short-term shared with Unified Runtime
        # The two plugins define a few things differrently so must
        # be built separately. This difference is spelled in
        # their "ur_bindings.hpp" files.
        #
        "../../../ur_bindings.hpp"
        "../../../pi2ur.hpp"
        "../../../pi2ur.cpp"
        "../../ur.hpp"
        "../../ur.cpp"
)

# set_target_properties(${TARGET_NAME} PROPERTIES
#     VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
#     SOVERSION "${PROJECT_VERSION_MAJOR}"
# )

target_include_directories(${TARGET_NAME}
  PRIVATE "${LEVEL_ZERO_INCLUDE_DIR}"
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
    LevelZeroLoader-Headers
    LevelZeroLoader
)

# if(UNIX)
#     set(GCC_COVERAGE_COMPILE_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden -fno-strict-aliasing")
#     set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
# endif()
