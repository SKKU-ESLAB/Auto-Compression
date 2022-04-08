#ifndef __CONFIG_H_
#define __CONFIG_H_

/////////////////- set unit size - ///////////////////
typedef uint16_t          unit_t;
//#define debug_mode
#define watch_pimindex    1
//////////////////////////////////////////////////////

#define NOP_END           111
#define EXIT_END          222

// SIZE IS BYTE
#define UNIT_SIZE         (int)(sizeof(unit_t))
#define WORD_SIZE         32
#define UNITS_PER_WORD    (WORD_SIZE / UNIT_SIZE)

#define GRF_SIZE          (8 * UNITS_PER_WORD * UNIT_SIZE)
#define SRF_SIZE          (8 * UNIT_SIZE)

enum class PIM_OPERATION {
    NOP = 0,
    JUMP,
    EXIT,
    MOV = 4,
    FILL,
    ADD = 8,
    MUL,
    MAC,
    MAD
};

enum class PIM_OP_TYPE {
    CONTROL = 0,
    DATA,
    ALU
};

enum class PIM_OPERAND {
    BANK = 0,
    GRF_A,
    GRF_B,
    SRF_A,
    SRF_M,
    NONE
};

#endif  // __CONFIG_H_
