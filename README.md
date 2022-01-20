# sample
grid_sample
# grid_sample #
```
mode: bilinear 
padding_mode: zeros
input: 512*960 
grid: 2048*3840*2 
output: 2048*3840 
```
1: download dpcpp-complier release version from https://github.com/intel/llvm/releases, and run unzip file :
``` :
tar -zxvf <file name>
```
 
2: compile code
```
cd dpcpp-compiler/dpcpp_compiler

source startup.sh

clang++ -fsycl  grid_sample_dpcpp.cpp -o grid_sample_dpcpp

```

3:run sample
```
./grid_sample_dpcpp
```

## Speed Test ##
### environment ###
```
Ubuntu 20.04 (GPU)
Windows10 (CPU)
clang version 14.0.0 (https://github.com/intel/llvm.git 739487cbb0994e006568a584dcf549ebb107bc11)

```
### kernel time ###
```
CPU : Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz
GPU : Intel(R) HD Graphics 530 
Average: average of 10 timers
```
|  | CPU | GPU |
| ------ | ------ | ------ |
| c++ | 842.667 msec | -- |
| dpc++ | 0.24894 msec | 0.1015 msec |

