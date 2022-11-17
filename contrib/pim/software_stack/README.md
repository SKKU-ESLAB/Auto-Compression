# PIM Software Stack
This is about `PIM Software Stack`

Contact me on [iESLAB GyungMo Kim](mailto:7bvcxz@gmail.com) for more information.

``` bash
git clone https://github.com/7bvcxz/PIM_SoftwareStack
cd PIM_SoftwareStack
```

## 간단 사용 설명서 (C-ONNX + PIM Software Stack)
``` bash
./build_so.sh   # (make clean && make so) → libpimss.so 생성
cp libpimss.so tutorial:
cd tutorial
./build.sh      # (gcc -o test test.c -L. -lpimss)
LD_LIBRARY_PATH=. ./test
```
기본 연산 테스트는 ADD로 설정되어 있습니다.  
test.c main함수의 주석을 통해 이외 연산들을 테스트해 볼 수 있습니다.  
또한, test.c의 각 연산함수(test_add_blas(), test_mul_blas(), test_gemv_blas())에서 m, n 값을 변환하여 다양한 연산크기로 수행할 수 있습니다.

## How to use PIM Software Stack
외부 c library에서 PIM Software Stack를 사용하는 예시는 tutorial 폴더의 test.c를 참고하시면 됩니다.
### Functions to call for PIM Computation in C Library
* 외부 c library →  PIM Software Stack 함수 호출 (아래의 순서로 호출)
  * blas_init(0);
    * PIM Software Stack Library를 Initialize합니다.
    * C Library Initialize시에 혹은 PIM 연산을 수행하기 전에 호출합니다.
  * 연산에 해당하는 PIM Preprocess 함수
    * PIM 연산에 필요한 데이터를 미리 추출하거나 PIM Memory의 적절한 위치에 저장합니다.
    * PIM BLAS 함수 이전에 호출합니다. (PIM 연산 시간에 미포함)
  * 연산에 해당하는 PIM BLAS 함수
    * PIM 연산을 수행합니다.
    * 연산에 필요한 메모리 커맨드를 PIM Memory(혹은 PIMsim / FPGA Emulator) 에게 전달합니다.
  * uint64_t time = pim_time();  #TODO#
    * PIM BLAS 함수를 통해 연산을 수행하는데 걸린 총 시간을 반환합니다. (단위 : ns)
 
* PIM Preprocess 함수 정의 (각각 Add, Mul, Gemv)
  * C_pimblasAddPreprocess(n, &in0, &in1);
    * n : Add를 수행할 vector의 길이 (element 개수)
    * in0, in1 : Add를 수행할 1-Dimension vector
    * 함수가 수행된 이후, in0, in1은 PIM Memory에 맵핑된 값으로 변합니다.
  * C_pimblasMulPreprocess(n, &in0, &in1);
    * 위와 동일합니다.
  * C_pimblasGemvPreprocess(m, n, &w);
    * m, n : 각각 GEMV를 수행할 Input, Output의 길이 (element 개수)
    * w : GEMV를 수행할 Weight 값. 2-Dimension Matrix를 1-Dimension Vector로 변환한 값입니다. W[i][j] == w[i x m + j]의 값을 가집니다.
    * 함수가 수행된 이후, w는 GEMV의 연산에 적합하도록 Padding된 이후 Transpose되어 PIM Memory에 맵핑되며, w는 해당 값으로 변합니다.
* PIM BLAS 함수 정의 (각각 Add, Mul, Gemv)
  * C_pim_add(n, in0, in1, out)
    * PIM으로 Add 연산을 수행하며, 연산 결과는 out에 저장됩니다.
  * C_pim_mul(n, in0, in1, out)
  * C_pim_gemv(m, n, in, w, out)
    * 위와 동일합니다.

### PIM Software Stack Install 모드
PIM Software Stack에서는 다양한 모듈과 연동하여 사용가능하도록 여러가지 모드로 Install할 수 있습니다.
1. PIM Software Stack + PIM Memory
2. PIM Software Stack + PIMsim (PIM Memory에 대한 Simulator)
3. PIM Software Stack + FPGA

어떤 모드로 Install할지 설정한 이후에 Install을 수행해야 합니다.

모드는 아래의 pim_config.h 주석을 통해서 설정할 수 있습니다.  
1번 모드는 실제 PIM Memory가 필요하기 때문에, 보통은 2번, FPGA가 존재할 시에는 3번 모드를 사용합니다.  
PIMsim은 PIM Software Stack이 생성한 메모리 커맨드에 따라 PIM Memory의 동작을 시뮬레이션하여 연산을 수행하고 연산 결과를 반환합니다.  
이를 통해 연산을 PIM Memory의 구조에 맞게 PIM Software Stack을 정확하게 설계했는지 확인할 수 있습니다.  
현재는 2번 모드를 사용하다가, 추후 FPGA 사용시에 3번 모드를 사용하여 실험해볼 수 있습니다.  
아래는 2번 모드로 Install하는 설정법입니다. (Default 설정되어 있음)  

``` cpp
// At pim_config.h
// #define fpga_mode
// #define debug_mode
#define compute_mode
```

### PIM Software Stack 주의사항
1. 현재, PIM Software Stack은 uint16_t 및 half float (16bit-float) 연산을 지원하고 있습니다. 그러나, 현재는 uint16_t로 빌드되어 있는 상태이며 추후 half float 연산으로도 빌드 가능하도록 빌드 옵션을 추가할 예정입니다.
2. ADD, MUL 연산의 경우 연산의 크기와 상관없이 PIM으로 연산을 정확히 수행함을 검증하였습니다. 그러나, GEMV 연산의 경우의 Output Size가 4096의 배수가 아닐 경우 오차가 발생하는 문제가 발생하고 있습니다. (현재 디버깅 진행중)

## Installing PIM Software Stack (C++ Library, C-ONNX는 해당 X)
* The implementation was compiled and tested with Ubuntu 18.04, gcc, g++ #TODO#

``` bash
./test.sh    # (make clean && make fpga && make)
./test gemv  # Test GEMV
./test add   # Test ADD
./test mul   # Test MUL
```

## PyTorch + PIM Software Stack
### Installing PyTorch + PIM Software Stack
* The implementation was compiled and tested with #TODO#
``` bash
python3 run.py
or
sudo python3 setup.py install
cp build/lib.linux-x86_64-3.6/* .")   # Different in other python versions (Need to check file name)
python3 app.py  # Run Test Code
```

### How To Use
Call pim library and use pim.to_pim(model)
``` python
import pim_library as pim

model = MyModel()
model = pim.to_pim(model)

output = model(input)
# It's over
```

## How PIM Software Stack Works
#TODO#


