#include "pim_utils.h"

PIM_OPERATION BitToPIM_OP(uint8_t *DataPtr)
{
    return (PIM_OPERATION)(DataPtr[3] >> 4);
}

PIM_OPERAND BitToDst(uint8_t *DataPtr)
{
    return (PIM_OPERAND)((DataPtr[3] >> 1) & 0b0000111);
}
PIM_OPERAND BitToSrc0(uint8_t *DataPtr)
{
    return (PIM_OPERAND)(((DataPtr[3] & 0b00000001) << 2) + (DataPtr[2] >> 6));
}

PIM_OPERAND BitToSrc1(uint8_t *DataPtr)
{
    return (PIM_OPERAND)((DataPtr[2] >> 3) & 0b00111);
}

int BitToDstIdx(uint8_t *DataPtr)
{
    return (int)(DataPtr[1] & 0b00000111);
}

int BitToSrc0Idx(uint8_t *DataPtr)
{
    return (int)((DataPtr[0] >> 4) & 0b0111);
}

int BitToSrc1Idx(uint8_t *DataPtr)
{
    return (int)(DataPtr[0] & 0b00000111);
}

int BitToSrc2Idx(uint8_t *DataPtr)
{
    return (int)(DataPtr[2] & 0b00000111);
}

int BitToImm0(uint8_t *DataPtr)
{
    int MSB = (DataPtr[2] >> 2) & 0b000001;
    MSB = ((MSB) ? (-1) : (1));
    int data = ((DataPtr[2] & 0b00000011) << 5) + (DataPtr[1] >> 3);
    return data * MSB;
}

int BitToImm1(uint8_t *DataPtr)
{
    return (int)(((DataPtr[1] & 0b00000001) << 7) + DataPtr[0]);
}

bool CheckAam(uint8_t *DataPtr)
{
    return (bool)(DataPtr[1] >> 7);
}

bool CheckReLU(uint8_t *DataPtr)
{
    return (bool)((DataPtr[1] >> 4) & 0b0001);
}

bool CheckDstFix(uint8_t *DataPtr)
{
    return (bool)((DataPtr[1] >> 3) & 0b00001);
}

bool CheckSrc0Fix(uint8_t *DataPtr)
{
    return (bool)(DataPtr[0] >> 7);
}

bool CheckSrc1Fix(uint8_t *DataPtr)
{
    return (bool)((DataPtr[0] >> 3) & 0b00001);
}
