#!/bin/bash
file="mbv2_r144.tflite"
cp -r ./tutorial/inference "./tutorial/${file}"
export PYTHONPATH=${PYTHONPATH}:$(pwd)
python3 examples/vww_patchbased.py
mkdir "./tutorial/${file}/Src/TinyEngine"
mv codegen "./tutorial/${file}/Src/TinyEngine"
cp -r ./TinyEngine/include "./tutorial/${file}/Src/TinyEngine"
cp -r ./TinyEngine/src "./tutorial/${file}/Src/TinyEngine"
mkdir "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp -r ./TinyEngine/third_party/CMSIS/CMSIS/NN/Include/*.h "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp -r ./TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/dsp "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp ./TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_common_tables.h "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp ./TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math_memory.h "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp ./TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math_types.h "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp ./TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math.h "./tutorial/${file}/Src/TinyEngine/include/arm_cmsis"
cp -r ./TinyEngine/third_party/CMSIS/CMSIS/Core/Include "./tutorial/${file}/Drivers/CMSIS"
