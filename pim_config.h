#ifndef __PIM_CONFIG_H_
#define __PIM_CONFIG_H_

#include <iostream>
#include <fcntl.h>		  // O_RDWR, O_SYNC
#include <sys/mman.h>	  // MAP_SHARED, PROT_READ
#include <random>		  // random_device
#include "half.hpp"

typedef uint16_t          unit_t;

// SIZE IS BYTE
#define UNIT_SIZE         (int)(sizeof(unit_t))
#define WORD_SIZE         32
#define UNITS_PER_WORD    (WORD_SIZE / UNIT_SIZE)
#define GRF_SIZE          (8 * UNITS_PER_WORD * UNIT_SIZE)
#define SRF_SIZE          (8 * UNIT_SIZE)

#define LEN_PIM			  0x100000000
#define BASE_PIM_ADDR	  0x140000000

enum class PIM_OP {
    ADD = 0,
    MUL,
    BN,
    GEMV,
    LSTM,
    RELU
};

class PIM_OP_ATTRS {
 public:
	 PIM_OP_ATTRS(){};
	 ~PIM_OP_ATTRS(){};

	 void ADD(uint8_t *x, uint8_t *y, uint8_t *z, int len);

};

class PIMKernel {
 public:
	uint32_t ukernel[32];

	void SetMicrokernelCode(PIM_OP op) {
		if (op == (PIM_OP::ADD)) {
			ukernel[0]  = 0b01000010000000001000000000000000; // MOV(AAM)  GRF_A  BANK
			ukernel[1]  = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[2]  = 0b10000010000010001000000000000000; // ADD(AAM)  GRF_A  BANK  GRF_A
			ukernel[3]  = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[4]  = 0b01000000010000001000000000000000; // MOV(AAM)  BANK   GRF_A
			ukernel[5]  = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[6]  = 0b01000010000000001000000000000000; // MOV(AAM)  GRF_A  BANK
			ukernel[7]  = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[8]  = 0b10000010000010001000000000000000; // ADD(AAM)  GRF_A  BANK  GRF_A
			ukernel[9]  = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[10] = 0b01000000010000001000000000000000; // MOV(AAM)  BANK   GRF_A
			ukernel[11] = 0b00010000000001000000100000000111; // JUMP      -1     7
			ukernel[12] = 0b00100000000000000000000000000000; // EXIT
		}
	}	
};

#endif  // __PIM_CONFIG_H_
