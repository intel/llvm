add_library(xptitest_subscriber SHARED XPTISubscriber.cpp)
target_compile_definitions(xptitest_subscriber PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(xptitest_subscriber PRIVATE xptifw)
