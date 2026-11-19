cmake_minimum_required(VERSION 3.20)
project(Skate3ClipStudio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CELLFORGE_BUILD_SDK_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_BINARY_DIR}/cellforge_mod_sdk")

cellforge_add_mod(skate3_clip_studio
  skate3_clip_studio.cpp
)
if(WIN32)
  target_link_libraries(skate3_clip_studio PRIVATE user32 gdi32 gdiplus shell32)
endif()
