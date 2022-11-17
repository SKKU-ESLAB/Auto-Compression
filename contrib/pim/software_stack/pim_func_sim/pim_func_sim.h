#ifndef __PIM_FUNC_SIM_H
#define __PIM_FUNC_SIM_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "pim_unit.h"
#include "../pim_config.h"
// #include "configuration.h"
// #include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

class PimFuncSim
{
public:
   PimFuncSim();
   void AddTransaction(uint64_t hex_addr, uint8_t *DataPtr, bool is_write);
   bool DebugModeFunc(uint64_t hex_addr);
   bool ModeChanger(uint64_t hex_addr);

   std::vector<std::string> bankmode;
   std::vector<bool> PIM_OP_MODE;
   std::vector<PimUnit *> pim_unit_;

   uint8_t *pmemAddr;
   uint64_t pmemAddr_size;
   unsigned int burstSize;

   uint64_t ReverseAddressMapping(Address &addr);
   uint64_t GetPimIndex(Address &addr);
   void PmemWrite(uint64_t hex_addr, uint8_t *DataPtr);
   void PmemRead(uint64_t hex_addr, uint8_t *DataPtr);
   void init(uint8_t *pmemAddr, uint64_t pmemAddr_size,
             unsigned int burstSize);
};

#ifdef __cplusplus
}
#endif


#endif // __PIM_FUNC_SIM_H
