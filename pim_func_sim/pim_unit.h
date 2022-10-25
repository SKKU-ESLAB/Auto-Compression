#ifndef __PIMUNIT_H_
#define __PIMUNIT_H_

#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <cmath>
#include "pim_func_config.h"
#include "pim_utils.h"
// #include "./configuration.h"
// #include "./common.h"
// #include "./half.hpp"

#ifdef __cplusplus
extern "C" {
#endif

class PimInstruction
{
public:
   PIM_OPERATION PIM_OP;
   PIM_OP_TYPE pim_op_type;
   bool is_aam;

   PIM_OPERAND dst;
   PIM_OPERAND src0;
   PIM_OPERAND src1;

   int dst_idx;
   int src0_idx;
   int src1_idx;
   int src2_idx;

   bool is_dst_fix;
   bool is_src0_fix;
   bool is_src1_fix;

   int imm0;
   int imm1;
};

class PimUnit
{
public:
   PimUnit(int id);
   int AddTransaction(uint64_t hex_addr, bool is_write, uint8_t *DataPtr);
   void SetSrf(uint64_t hex_addr, uint8_t *DataPtr);
   void SetGrf(uint64_t hex_addr, uint8_t *DataPtr);
   void SetCrf(uint64_t hex_addr, uint8_t *DataPtr);
   void init(uint8_t *pmemAddr, uint64_t pmemAddr_size,
             unsigned int burstSize);
   bool DebugMode();
   void PrintPIM_IST(PimInstruction inst);
   void PrintOperand(int op_id);

   void PushCrf(int CRF_idx, uint8_t *DataPtr);
   void SetOperandAddr(uint64_t hex_addr);
   void Execute();
   void _ADD();
   void _MUL();
   void _MAC();
   void _MAD();
   void _MOV();

   PimInstruction CRF[32];
   uint8_t PPC;
   int LC;

   int pim_id;
   int debug_cnt;

   unit_t *GRF_A_;
   unit_t *GRF_B_;
   unit_t *SRF_A_;
   unit_t *SRF_M_;

   unit_t *dst;
   unit_t *src0;
   unit_t *src1;
   unit_t *bank_data_;

   uint8_t *pmemAddr_;
   uint64_t pmemAddr_size_;
   unsigned int burstSize_;
};

#ifdef __cplusplus
}
#endif

#endif // PIMUNIT_H_
