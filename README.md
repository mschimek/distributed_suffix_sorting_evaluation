# Fast and Lightweight Distributed Suffix Array Construction

This repository contains the source codes used for the evaluation part of the paper "Fast and Lightweight Distributed Suffix Array Construction".

## Dependencies

We tested the following software stack to compile this code:
- GCC 12.2.0
- IntelMPI 2021.11
- (you also require Intel TBB in a recent version)

Note: This is a collection of several projects. Every subdirectory includes its own license, which applies to the code within that folder.

## Build and run our lightweight DCX implementation

To compile the algorithm, you can run:
```bash
cd lightweight_dcx
cmake -B build -DCMAKE_BUILD_TYPE=Release -DINCLUDE_ALL_SORTERS=ON -DOPTIMIZE_DATA_TYPES=OFF -DIPS4O_DISABLE_PARALLEL=ON
cmake --build build --parallel --target cli
```

Note that by default only the DC39 variant using Packing is compiled (due to long compile times).
Setting the cmake option `DCX_BUILD_ALL` also compiles DCX variants for other X.

For executing the code, run
```bash
mpiexec -n <number ranks> ./build/cli --input <file> --textsize <size in bytes> --dcx <dc39> --atomic-sorter ams --discarding-threshold 0.7 --ams-levels 2 --splitter-sampling random --splitter-sorting central --use-random-sampling-splitters --num-samples-splitters 20000 --buckets-sample-phase 16,16 --buckets-merging-phase 64,64,16 --use-binary-search-for-splitters --use-randomized-chunks --avg-chunks-pe 10000 --use-char-packing-samples --use-char-packing-merging --buckets-phase3 1 --samples-buckets-phase3 10000 --rearrange-buckets-balanced --use-compressed-buckets --pack-extra-words 0 --json-output-path <output-path-for-logs>
```

On small scale using fewer buckets might be beneficial.
The main repository can be found here:
https://github.com/mschimek/distributed_suffix_sorting/


## Build and run (extended) PSAC

To compile the algorithm, you can run:
```bash
cd psac
cmake -B build -DCMAKE_BUILD_TYPE=Release  -DIPS4O_DISABLE_PARALLEL=ON
cmake --build build --parallel --target psac
```
For compiling our 40-bit integer variant set the cmake option `USE_40_BIT_INDEX`.
For executing the code, run
```bash
mpiexec -n <number ranks> ./build/bin/psac --file <file> --textsize <size in bytes> --threshold_fast_resolval <0.1, 1.0>  --json_output_path <output-path-for-logs>  [--use_ams --use_ips4o  --ams_levels <1, 2>]
```
Source:
Flick, Patrick and Srinivas Aluru. "Parallel distributed memory construction of suffix and longest common prefix arrays." Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, ACM, 2015.
https://github.com/patflick/psac


## Build and run (extended) dPD

To compile the algorithm, you can run:
```bash
cd dPD
cmake -B build -DCMAKE_BUILD_TYPE=Release  -DIPS4O_DISABLE_PARALLEL=ON
cmake --build build --parallel --target dsaca
```
For executing the code, run
```bash
mpiexec -n <number ranks> ./build/benchmark/dsaca --input <file> --size <size in bytes> --discarding [--use_ams --ams_levels <1,2>]
```
Source:
Johannes Fischer and Florian Kurpicz. "Lightweight Distributed Suffix Array Construction." Proceedings of the Twenty-First Workshop on Algorithm Engineering and Experiments ({ALENEX}), SIAM, 2019
https://github.com/kurpicz/dsss

## Build and run the DC3/7/13 implementation of T. Bingmann

To compile the algorithm, you can run:
```bash
cd B_DC3_7_13
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target pDCX
```

For executing the code, run
```bash
mpiexec -n <number ranks> ./build/src/pDCX --input <file> --textsize <size in bytes> --dcx <3/7/13>
```

Source:
Timo Bingmann: https://github.com/bingmann/pDCX

## Build and run libsais

To compile the algorithm, you can run:
```bash
cd libsais
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLIBSAIS_USE_OPENMP=ON
cmake --build build --parallel
```

For executing the code, run
```bash
OMP_PLACES=cores OMP_PROC_BIND=close ./build/run_libsais --input <file> --textsize <size in bytes> --num_threads <num threads to use>  --json_output_path bla --iteration <used to differtiate multiple runs, directly written to output>
```

Source:
Ilya Grenov: https://github.com/IlyaGrebnov/libsais


## Remarks
To reduce the memory footprint of IntelMPI on our system, we set the following environment variables (suggested by the SuperMUC-NG operations team) in our evaluation

```
export I_MPI_SHM_CELL_FWD_NUM=0
export I_MPI_SHM_CELL_EXT_NUM_TOTAL=0
export I_MPI_SHM_CELL_BWD_SIZE=65536
export I_MPI_SHM_CELL_BWD_NUM=64
export I_MPI_MALLOC=0
export I_MPI_SHM_HEAP_VSIZE=0
```
