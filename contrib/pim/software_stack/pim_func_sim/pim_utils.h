#ifndef __UTILS_H_
#define __UTILS_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "pim_func_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ReLU(x) ((x < 0) ? (0) : (x))

PIM_OPERATION BitToPIM_OP(uint8_t *DataPtr);
PIM_OPERAND BitToDst(uint8_t *DataPtr);
PIM_OPERAND BitToSrc0(uint8_t *DataPtr);
PIM_OPERAND BitToSrc1(uint8_t *DataPtr);
int BitToDstIdx(uint8_t *DataPtr);
int BitToSrc0Idx(uint8_t *DataPtr);
int BitToSrc1Idx(uint8_t *DataPtr);
int BitToSrc2Idx(uint8_t *DataPtr);
int BitToImm0(uint8_t *DataPtr);
int BitToImm1(uint8_t *DataPtr);
bool CheckAam(uint8_t *DataPtr);
bool CheckDstFix(uint8_t *DataPtr);
bool CheckSrc0Fix(uint8_t *DataPtr);
bool CheckSrc1Fix(uint8_t *DataPtr);
bool CheckReLU(uint8_t *DataPtr);

#ifdef __cplusplus
}
#endif

#endif // __UTILS_H_
