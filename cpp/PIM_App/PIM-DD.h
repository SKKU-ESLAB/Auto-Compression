/******************************************************************************
*  \file		PIM-DD.c
*	
*  \details		PIM Linux Device Driver
*
*  \author		Seongjin LEE
*
*  \Tested		with linux kernel 5.10.43 on gem5
*
*******************************************************************************/

/******************************************************************************
HBM-PIM address mapping
32       31 ... 18 17 ... 13   12 ... 9  8 ... 5      4 ... 0
SID(1b), row(14b), column(5b), bank(4b), channel(4b), offset(5b)
*******************************************************************************/

#define SBMR_ROW_ADDR 0x3FFF
#define ABMR_ROW_ADDR 0x3FFE
#define PIM_OP_MODE_ROW_ADDR 0x3FFD
#define CRF_ROW_ADDR 0x3FFC
#define GRF_ROW_ADDR 0x3FFB
#define SRF_ROW_ADDR 0x3FFA

#define GET_SBMR_ADDR(BANK, CH) 1<<32 | SBMR_ROW_ADDR<<18 |  BANK<<9 | CH<<5
#define GET_ABMR_ADDR(BANK, CH) 1<<32 | ABMR_ROW_ADDR<<18 |  BANK<<9 | CH<<5
#define GET_PIM_OP_MODE_ADDR(BANK, CH) 1<<32 | PIM_OP_MODE_ROW_ADDR<<18 |  BANK<<9 | CH<<5
#define GET_CRF_ADDR(BANK, CH) 1<<32 | CRF_ROW_ADDR<<18 |  BANK<<9 | CH<<5
#define GET_GRF_ADDR(BANK, CH) 1<<32 | GRF_ROW_ADDR<<18 |  BANK<<9 | CH<<5
#define GET_SRF_ADDR(BANK, CH) 1<<32 | SRF_ROW_ADDR<<18 |  BANK<<9 | CH<<5

//#define BASE_PIM_ADDR 0x23FE80000 // GET_SRF_ADDR(0,0) + 0x40000000
#define BASE_PIM_ADDR 0x140000000 // 0x100000000(4GB) + 0x40000000(1GB)
//#define LEN_PIM 0x180000 // 1.5MB size
#define LEN_PIM 0x100000000 // 4GB size
