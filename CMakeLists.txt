set(VTK_DIR /usr/local/)

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

#-----------------------------------------------------
project(MandelbrotExplorer LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

find_package(VTK CONFIG)
find_package(VTK REQUIRED)

if (NOT VTK_FOUND)
  message("Skipping Mandelbrot: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()

message (STATUS "VTK_VERSION: ${VTK_VERSION}")
message (STATUS "VTK_LIBRARIES: ${VTK_LIBRARIES}")

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_C_FLAGS "-O0 -ggdb")
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_C_FLAGS_RELEASE "-O0 -ggdb")
set(CMAKE_CXX_FLAGS "-O0 -ggdb -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O0 -ggdb -std=c++14")

set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -arch=sm_37 -c -std=c++14")

include_directories(
		"/usr/local/include/vtk-9.0/"
)

link_directories(
		"/usr/local/lib/"
		"/usr/lib/x86_64-linux-gnu/"
)

# set(LD_LIBRARY_PATH "/usr/local/lib/")

add_executable(Mandelbrot Mandelbrot.cu )
target_link_libraries(Mandelbrot ${VTK_LIBRARIES})

# Turn this on for memory leak debugging
set(VTK_DEBUG_LEAKS ON)

vtk_module_autoinit(
    TARGETS Mandelbrot
    MODULES ${VTK_LIBRARIES} 
) 

