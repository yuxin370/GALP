# G-ALP

A heavily modified, standalone-friendly fork of the original G-ALP codebase:
https://github.com/cwida/FastLanesGpu-Damon2025

---
## Project layout

* `src/` — core library sources and public headers
* `external/` — vendored third-party code (ALP, generated pack/unpack, headers)
* `benchmark/` — benchmark executables and generated bindings
* `test/` — unit and integration tests

---

## Start up

### Prerequisites

- CMake ≥ 3.22
- An NVIDIA GPU with CUDA support + a working CUDA Toolkit installation
- `nvcc` available in your `PATH`
- **nvCOMP** installed and discoverable by CMake
- A C++20 compiler (Clang recommended)

---

## Build

From the `galp/` directory:

```bash
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGALP_BUILD_TESTS=ON \
  -DGALP_BUILD_BENCHMARKS=ON

cmake --build . -j
````

---

## Run benchmarks

After building with `-DGALP_BUILD_BENCHMARKS=ON`, executables are produced under `build/benchmark/`. For example:

```bash
./benchmark/micro-benchmarks u64 decompress 4 1 stateful-branchless none 32 32 200 200 4 10 1
```

---


## Notes on dependencies

### nvCOMP

Ensure your nvCOMP installation provides a CMake package configuration (so `find_package(nvcomp CONFIG REQUIRED)` works) and that CMake can locate it (e.g., via default system paths or `CMAKE_PREFIX_PATH`).

### Thrust / CCCL

The build prefers the Thrust headers shipped with the CUDA Toolkit. A recent Thrust version is required.
