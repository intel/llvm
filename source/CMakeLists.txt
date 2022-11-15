set(TARGET_NAME ur_test)

add_library(${TARGET_NAME}
    STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_api.cpp
)

target_link_libraries(${TARGET_NAME}
    ${CMAKE_DL_LIBS}
)
