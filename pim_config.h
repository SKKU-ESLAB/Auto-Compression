#ifndef __PIM_CONFIG_H_
#define __PIM_CONFIG_H_

#include <iostream>
#include <fcntl.h>		  // O_RDWR, O_SYNC
#include <sys/mman.h>	  // MAP_SHARED, PROT_READ
#include <random>		  // random_device
#include "half.hpp"

typedef uint16_t			unit_t;

// SIZE IS BYTE
#define UNIT_SIZE			(int)(sizeof(unit_t))
#define WORD_SIZE			32
#define UNITS_PER_WORD		(WORD_SIZE / UNIT_SIZE)
#define GRF_SIZE			(8 * UNITS_PER_WORD * UNIT_SIZE)
#define SRF_SIZE			(8 * UNIT_SIZE)

#define LEN_PIM				0x100000000
#define BASE_PIM_ADDR		0x140000000

#define EVEN_BANK 0
#define ODD_BANK 1

#define NUM_WORD_PER_ROW	  32
#define NUM_UNIT_PER_WORD	  16
#define	NUM_CHANNEL			  16
#define NUM_BANK_PER_CHANNEL  16
#define NUM_BANK			  (NUM_BANK_PER_CHANNEL * NUM_CHANNEL)
#define SIZE_WORD			  32
#define SIZE_ROW			  (SIZE_WORD * NUM_WORD_PER_ROW)

#define MAP_SBMR             0x3fff
#define MAP_ABMR             0x3ffe
#define MAP_PIM_OP_MODE      0x3ffd
#define MAP_CRF              0x3ffc
#define MAP_GRF              0x3ffb
#define MAP_SRF              0x3ffa

struct Address {
    Address()
        : channel(-1), rank(-1), bankgroup(-1), bank(-1), row(-1), column(-1) {}
    Address(int channel, int rank, int bankgroup, int bank, int row, int column)
        : channel(channel),
          rank(rank),
          bankgroup(bankgroup),
          bank(bank),
          row(row),
          column(column) {}
    Address(const Address& addr)
        : channel(addr.channel),
          rank(addr.rank),
          bankgroup(addr.bankgroup),
          bank(addr.bank),
          row(addr.row),
          column(addr.column) {}
    int channel;
    int rank;
    int bankgroup;
    int bank;
    int row;
    int column;
};

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
	void MUL(uint8_t *x, uint8_t *y, uint8_t *z, int len);
	void BN(uint8_t *x, uint8_t *y, uint8_t *z, int len);
	void GEMV(uint8_t *y, uint8_t *z, int len_in, int len_out);
	void LSTM(uint8_t *x, uint8_t *y, uint8_t *z, int len);

	int len_in;
	int len_out;
	PIM_OP pim_op;
};

class PIMKernel {
 public:
	PIM_OP pim_op;
	uint32_t ukernel[32];
    uint32_t ukernel_extra[32];

	void SetMicrokernelCode(PIM_OP op) {
		if (op == (PIM_OP::ADD)) {
			ukernel[0]  = 0b01000010000000001000000000000000; // MOV(A)  GRF_A[A0]  BANK
			ukernel[1]  = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[2]  = 0b10000010000010001000000000000000; // ADD(A)  GRF_A[A0]  BANK      GRF_A[A0]
			ukernel[3]  = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[4]  = 0b01000000010000001000000000000000; // MOV(A)  BANK       GRF_A[A0]
			ukernel[5]  = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[6]  = 0b01000010000000001000000000000000; // MOV(A)  GRF_A[A0]  BANK
			ukernel[7]  = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[8]  = 0b10000010000010001000000000000000; // ADD(A)  GRF_A[A0]  BANK      GRF_A[A0]
			ukernel[9]  = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[10] = 0b01000000010000001000000000000000; // MOV(A)  BANK       GRF_A[A0]
			ukernel[11] = 0b00010000000001000000100000000111; // JUMP    -1         7
			ukernel[12] = 0b00100000000000000000000000000000; // EXIT
			pim_op = PIM_OP::ADD;
		}
		else if (op == (PIM_OP::MUL)) {
			ukernel[0]  = 0b01000010000000001000000000000000; // MOV(A)  GRF_A[A0]  BANK
            ukernel[1]  = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[2]  = 0b10010010000010001000000000000000; // MUL(A)  GRF_A[A0]  BANK      GRF_A[A0]
            ukernel[3]  = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[4]  = 0b01000000010000001000000000000000; // MOV(A)  BANK       GRF_A[A0]
            ukernel[5]  = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[6]  = 0b01000010000000001000000000000000; // MOV(A)  GRF_A[A0]  BANK
            ukernel[7]  = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[8]  = 0b10010010000010001000000000000000; // MUL(A)  GRF_A[A0]  BANK      GRF_A[A0]
            ukernel[9]  = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[10] = 0b01000000010000001000000000000000; // MOV(A)  BANK       GRF_A[A0]
            ukernel[11] = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[12] = 0b00100000000000000000000000000000; // EXIT
			pim_op = PIM_OP::MUL;
		}
		else if (op == (PIM_OP::BN)) {
			ukernel[0]  = 0b01000010000000001000000000000000;  // MOV(A)  GRF_A[A0] BANK
			ukernel[1]  = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[2]  = 0b10010010000010001000000000000000;  // MUL(A)  GRF_A[A0] BANK      GRF_A[A0]
			ukernel[3]  = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[4]  = 0b10000010000010001000000000000000;  // ADD(A)  GRF_A[A0] BANK      GRF_A[A0]
			ukernel[5]  = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[6]  = 0b01000000010000001000000000000000;  // MOV(A)  BANK      GRF_A[A0]
			ukernel[7]  = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[8]  = 0b01000010000000001000000000000000;  // MOV(A)  GRF_A[A0] BANK
			ukernel[9]  = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[10] = 0b10010010000010001000000000000000;  // MUL(A)  GRF_A[A0] BANK      GRF_A[A0]
			ukernel[11] = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[12] = 0b10000010000010001000000000000000;  // ADD(A)  GRF_A[A0] BANK      GRF_A[A0]
			ukernel[13] = 0b00010000000001000000100000000111;  // JUMP    -1        7 
			ukernel[14] = 0b01000000010000001000000000000000;  // MOV(A)  BANK      GRF_A[A0]
			ukernel[15] = 0b00010000000001000000100000000111;  // JUMP    -1        7
			ukernel[16] = 0b00100000000000000000000000000000;  // EXIT
			pim_op = PIM_OP::BN;
		}
		else if (op == (PIM_OP::GEMV)) {
	        ukernel[0] = 0b10100100001000001000100000000000; // MAC(A)  GRF_B[A0]  BANK      SRF_M[A0]
            ukernel[1] = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[2] = 0b00100000000000000000000000000000; // EXIT

            ukernel_extra[0] = 0b10100100001000001000100000000000; // MAC(A)  GRF_B[A0]  BANK      SRF_M[A0]
            ukernel_extra[1] = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel_extra[2] = 0b01000000100000000000000000000000; // MOV     BANK       GRF_B[0]
            ukernel_extra[3] = 0b00100000000000000000000000000000; // EXIT
			pim_op = PIM_OP::GEMV;
		}
		else if (op == (PIM_OP::LSTM)) {
            ukernel[0] = 0b10100100001000001000100000000000; // MAC(A)  GRF_B[0]  BANK  SRF_M[A0]
            ukernel[1] = 0b00010000000001000000100000000111; // JUMP    -1         7
            ukernel[2] = 0b00100000000000000000000000000000; // EXIT

    		ukernel_extra[0] = 0b10000010100000000000100010000000; // ADD     GRF_A[0]  GRF_B[0]  BANK
    		ukernel_extra[1] = 0b10000100010000000000100110000000; // ADD     GRF_B[1]  GRF_A[0]  BANK
    		ukernel_extra[2] = 0b01000000100000000000000000010000; // MOV     BANK      GRF_B[1]
    		ukernel_extra[3] = 0b00100000000000000000000000000000; // EXIT
			pim_op = PIM_OP::LSTM;
		}
	}	
};

#endif  // __PIM_CONFIG_H_
