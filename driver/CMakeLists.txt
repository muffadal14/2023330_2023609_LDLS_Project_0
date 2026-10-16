cmake_minimum_required(VERSION 3.14)

# Build the driver against the llama.cpp checkout and its existing build tree.
# Nothing here modifies llama.cpp; the driver only links the libraries it
# already produces.
project(llama_latex_driver CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED LLAMA_ROOT)
    set(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
endif()
if (NOT DEFINED LLAMA_BUILD_DIR)
    set(LLAMA_BUILD_DIR "${LLAMA_ROOT}/build")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Threads REQUIRED)

# Import the prebuilt llama.cpp import libraries.  A glob finds them whatever
# the per-config subdirectory is called, so the driver builds with Ninja and
# with Visual Studio alike.
set(LLAMA_LIBS llama llama-common llama-common-base ggml ggml-base ggml-cpu)

add_library(llama_driver_deps INTERFACE)

foreach(lib ${LLAMA_LIBS})
    file(GLOB_RECURSE found "${LLAMA_BUILD_DIR}/*/${lib}.lib" "${LLAMA_BUILD_DIR}/${lib}.lib")
    if (NOT found)
        message(FATAL_ERROR
            "Could not find ${lib}.lib under ${LLAMA_BUILD_DIR}. Build llama.cpp "
            "first, or point LLAMA_BUILD_DIR at its build directory.")
    endif()
    list(GET found 0 lib_path)
    target_link_libraries(llama_driver_deps INTERFACE "${lib_path}")
endforeach()

target_include_directories(llama_driver_deps INTERFACE
    "${LLAMA_ROOT}/include"
    "${LLAMA_ROOT}/common"
    "${LLAMA_ROOT}/ggml/include"
    "${LLAMA_ROOT}/vendor"
    "${LLAMA_ROOT}/ggml/src")

target_link_libraries(llama_driver_deps INTERFACE Threads::Threads)

add_executable(llama-latex-driver latex_driver.cpp)
target_link_libraries(llama-latex-driver PRIVATE llama_driver_deps)