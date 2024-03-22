add_library(moondream OBJECT
            moondream.cpp
            moondream.h
            vision_encoder.cpp
            vision_encoder.h
            )

target_link_libraries(moondream PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(moondream PUBLIC .)
target_include_directories(moondream PUBLIC ../..)
target_include_directories(moondream PUBLIC ../../common)

target_compile_features(moondream PRIVATE cxx_std_11)

add_library(moondream_static STATIC $<TARGET_OBJECTS:moondream>)
if (BUILD_SHARED_LIBS)
    set_target_properties(moondream PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(moondream PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(moondream_shared SHARED $<TARGET_OBJECTS:moondream>)
    target_link_libraries(moondream_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS mondream_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(moondream PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(moondream BUILD_INFO)
endif()

set(TARGET md-cli)
add_executable(md-cli md-cli.cpp)
install(TARGETS md-cli RUNTIME)
target_link_libraries(md-cli PRIVATE common moondream ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(moondream PRIVATE cxx_std_11)
