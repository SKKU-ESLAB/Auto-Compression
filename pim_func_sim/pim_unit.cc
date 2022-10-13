#include "pim_unit.h"

using half_float::half;

PimUnit::PimUnit(int id)
    : pim_id(id)
{
    PPC = 0; // PIM program counter : Points the PIM Instruction to execute in
             //                       CRF register
    LC = 0;  // Loop counter : A counter to perform NOP, JUMP Instructions

    // Initialize PIM Registers
    GRF_A_ = (unit_t *)malloc(GRF_SIZE);
    GRF_B_ = (unit_t *)malloc(GRF_SIZE);
    SRF_A_ = (unit_t *)malloc(SRF_SIZE);
    SRF_M_ = (unit_t *)malloc(SRF_SIZE);
    bank_data_ = (unit_t *)malloc(WORD_SIZE);
    dst = (unit_t *)malloc(WORD_SIZE);

    for (int i = 0; i < WORD_SIZE / (int)sizeof(unit_t); i++)
        dst[i] = 0;
    for (int i = 0; i < GRF_SIZE / (int)sizeof(unit_t); i++)
    {
        GRF_A_[i] = 0;
        GRF_B_[i] = 0;
    }
    for (int i = 0; i < SRF_SIZE / (int)sizeof(unit_t); i++)
    {
        SRF_A_[i] = 0;
        SRF_M_[i] = 0;
    }
}

void PimUnit::init(uint8_t *pmemAddr, uint64_t pmemAddr_size,
                   unsigned int burstSize)
{
    pmemAddr_ = pmemAddr;
    pmemAddr_size_ = pmemAddr_size;
    burstSize_ = burstSize;
}

// Return to print out debugging information or not
//  Can set debug_mode and watch_pimindex at pim_config.h
bool PimUnit::DebugMode()
{
#ifndef debug_mode
    return false;
#endif

    if (pim_id == watch_pimindex)
        return true;
    return false;
}

// Print operand's type (register type, Bank)
void PimUnit::PrintOperand(int op_id)
{
    if (op_id == 0)
        std::cout << "BANK";
    else if (op_id == 1)
        std::cout << "GRF_A";
    else if (op_id == 2)
        std::cout << "GRF_B";
    else if (op_id == 3)
        std::cout << "SRF_A";
    else if (op_id == 4)
        std::cout << "SRF_M";
}

// Print Instruction according to PIM operation id
void PimUnit::PrintPIM_IST(PimInstruction inst)
{
    if (inst.PIM_OP == (PIM_OPERATION)0)
        std::cout << "NOP\t";
    else if (inst.PIM_OP == (PIM_OPERATION)1)
        std::cout << "JUMP\t";
    else if (inst.PIM_OP == (PIM_OPERATION)2)
        std::cout << "EXIT\t";
    else if (inst.PIM_OP == (PIM_OPERATION)4)
        std::cout << "MOV\t";
    else if (inst.PIM_OP == (PIM_OPERATION)5)
        std::cout << "FILL\t";
    else if (inst.PIM_OP == (PIM_OPERATION)8)
        std::cout << "ADD\t";
    else if (inst.PIM_OP == (PIM_OPERATION)9)
        std::cout << "MUL\t";
    else if (inst.PIM_OP == (PIM_OPERATION)10)
        std::cout << "MAC\t";
    else if (inst.PIM_OP == (PIM_OPERATION)11)
        std::cout << "MAD\t";

    if (inst.pim_op_type == (PIM_OP_TYPE)0)
    { // CONTROL
        std::cout << (int)inst.imm0 << "\t";
        std::cout << (int)inst.imm1 << "\t";
    }
    else if (inst.pim_op_type == (PIM_OP_TYPE)1)
    { // DATA
        PrintOperand((int)inst.dst);
        if ((int)inst.dst != 0)
        {
            if (inst.is_aam == 0 || inst.is_dst_fix)
                std::cout << "[" << inst.dst_idx << "]";
            else
                std::cout << "(A)";
        }
        std::cout << "  ";

        PrintOperand((int)inst.src0);
        if ((int)inst.src0 != 0)
        {
            if (inst.is_aam == 0 || inst.is_src0_fix)
                std::cout << "[" << inst.src0_idx << "]";
            else
                std::cout << "(A)";
        }
        std::cout << "  ";
    }
    else if (inst.pim_op_type == (PIM_OP_TYPE)2)
    { // ALU
        PrintOperand((int)inst.dst);
        if ((int)inst.dst != 0)
        {
            if (inst.is_aam == 0 || inst.is_dst_fix)
                std::cout << "[" << inst.dst_idx << "]";
            else
                std::cout << "(A)";
        }
        std::cout << "  ";
        PrintOperand((int)inst.src0);
        if ((int)inst.src0 != 0)
        {
            if (inst.is_aam == 0 || inst.is_src0_fix)
                std::cout << "[" << inst.src0_idx << "]";
            else
                std::cout << "(A)";
        }
        std::cout << "  ";
        PrintOperand((int)inst.src1);
        if ((int)inst.src1 != 0)
        {
            if (inst.is_aam == 0 || inst.is_src1_fix)
                std::cout << "[" << inst.src1_idx << "]";
            else
                std::cout << "(A)";
        }
        std::cout << "  ";
    }
    std::cout << "\n";
}

// Set pim_unit's SRF Register
//  Data in DataPtr will be 32-byte data
//  Front 16-byte of DataPtr is written to SRF_A Register and
//  next 16-byte of DataPtr is written to SRF_M Register
void PimUnit::SetSrf(uint64_t hex_addr, uint8_t *DataPtr)
{
    if (DebugMode())
        std::cout << " PU: SetSrf\n";
    memcpy(SRF_A_, DataPtr, SRF_SIZE);
    memcpy(SRF_M_, DataPtr + SRF_SIZE, SRF_SIZE);
}

// Set pim_unit's GRF Register
//  Data in DataPtr will be 32-byte bata
//  if hex_addr.Column Address is 0~7
//  Column Address 0~7 data is written to GRF_A 0~7 each
//  if hex_addr.Column Address is 8~15
//  Column Address 8~15 data is written to GRF_B 0~7 each
void PimUnit::SetGrf(uint64_t hex_addr, uint8_t *DataPtr)
{
    if (DebugMode())
        std::cout << "  PU: SetGrf\n";
    Address addr = AddressMapping(hex_addr);
    if (addr.column < 8)
    { // GRF_A
        unit_t *target = GRF_A_ + addr.column * WORD_SIZE / sizeof(unit_t);
        memcpy(target, DataPtr, WORD_SIZE);
    }
    else
    { // GRF_B
        GRF_B_[15] = 0;
        unit_t *target = GRF_B_ + (addr.column - 8) * WORD_SIZE / sizeof(unit_t);
        memcpy(target, DataPtr, WORD_SIZE);
    }
}

// Set pim_unit's GRF Register
//  Column Address 0 data is written to CRF 0~7
//  Column Address 1 data is written to CRF 8~15
//  Column Address 2 data is written to CRF 16~23
//  Column Address 3 data is written to CRF 24~31
void PimUnit::SetCrf(uint64_t hex_addr, uint8_t *DataPtr)
{
    if (DebugMode())
        std::cout << "  PU: SetCrf\n";
    Address addr = AddressMapping(hex_addr);
    int CRF_idx = addr.column * 8;
    for (int i = 0; i < 8; i++)
    {
        PushCrf(CRF_idx + i, DataPtr + 4 * i);
    }
}

// Map 32-bit data into structure of PIM_INSTRUCTION
void PimUnit::PushCrf(int CRF_idx, uint8_t *DataPtr)
{
    CRF[CRF_idx].PIM_OP = BitToPIM_OP(DataPtr);
    CRF[CRF_idx].is_aam = CheckAam(DataPtr);
    CRF[CRF_idx].is_dst_fix = CheckDstFix(DataPtr);
    CRF[CRF_idx].is_src0_fix = CheckSrc0Fix(DataPtr);
    CRF[CRF_idx].is_src1_fix = CheckSrc1Fix(DataPtr);

    switch (CRF[CRF_idx].PIM_OP)
    {
    case PIM_OPERATION::ADD:
    case PIM_OPERATION::MUL:
    case PIM_OPERATION::MAC:
    case PIM_OPERATION::MAD:
        CRF[CRF_idx].pim_op_type = PIM_OP_TYPE::ALU;
        CRF[CRF_idx].dst = BitToDst(DataPtr);
        CRF[CRF_idx].src0 = BitToSrc0(DataPtr);
        CRF[CRF_idx].src1 = BitToSrc1(DataPtr);
        CRF[CRF_idx].dst_idx = BitToDstIdx(DataPtr);
        CRF[CRF_idx].src0_idx = BitToSrc0Idx(DataPtr);
        CRF[CRF_idx].src1_idx = BitToSrc1Idx(DataPtr);
        break;
    case PIM_OPERATION::MOV:
    case PIM_OPERATION::FILL:
        CRF[CRF_idx].pim_op_type = PIM_OP_TYPE::DATA;
        CRF[CRF_idx].dst = BitToDst(DataPtr);
        CRF[CRF_idx].src0 = BitToSrc0(DataPtr);
        CRF[CRF_idx].src1 = BitToSrc1(DataPtr);
        CRF[CRF_idx].dst_idx = BitToDstIdx(DataPtr);
        CRF[CRF_idx].src0_idx = BitToSrc0Idx(DataPtr);
        break;
    case PIM_OPERATION::NOP:
        CRF[CRF_idx].pim_op_type = PIM_OP_TYPE::CONTROL;
        CRF[CRF_idx].imm1 = BitToImm1(DataPtr);
        break;
    case PIM_OPERATION::JUMP:
        CRF[CRF_idx].imm0 = CRF_idx + BitToImm0(DataPtr);
        CRF[CRF_idx].imm1 = BitToImm1(DataPtr);
        CRF[CRF_idx].pim_op_type = PIM_OP_TYPE::CONTROL;
        break;
    case PIM_OPERATION::EXIT:
        CRF[CRF_idx].pim_op_type = PIM_OP_TYPE::CONTROL;
        break;
    default:
        break;
    }
    if (DebugMode())
    {
        std::cout << "  PU: program  ";
        PrintPIM_IST(CRF[CRF_idx]);
    }
}

// Execute PIM_INSTRUCTIONS in CRF register and compute PIM
int PimUnit::AddTransaction(uint64_t hex_addr, bool is_write,
                            uint8_t *DataPtr)
{
    // Read data from physical memory
    if (!is_write)
        memcpy(bank_data_, pmemAddr_ + hex_addr, WORD_SIZE);

    // Map operand data's offset to computation pointers properly
    SetOperandAddr(hex_addr);

    // Execute PIM_INSTRUCTION
    // Is executed using computation pointers mapped from SetOperandAddr
    Execute();

    // if PIM_INSTRUCTION that writes data to physical memory
    // is executed, write to physcial memory
    if (CRF[PPC].PIM_OP == PIM_OPERATION::MOV &&
        CRF[PPC].dst == PIM_OPERAND::BANK)
    {
        memcpy(pmemAddr_ + hex_addr, dst, WORD_SIZE);
    }

    // Point to next PIM_INSTRUCTION
    PPC += 1;

    // Deal with PIM operation NOP & JUMP
    //  Performed by using LC(Loop Counter)
    //  LC copies the number of iterations and gets lower by 1 when executed
    //  Repeats until LC gets to 1 and escapes the iteration
    if (CRF[PPC].PIM_OP == PIM_OPERATION::NOP)
    {
        if (LC == 0)
        {
            LC = CRF[PPC].imm1;
        }
        else if (LC > 1)
        {
            LC -= 1;
        }
        else if (LC == 1)
        {
            PPC += 1;
            LC = 0;
            return NOP_END;
        }
        if (DebugMode())
        {
            std::cout << "  PU: NOP left (" << LC << ")\n";
        }
        return 0;
    }
    else if (CRF[PPC].PIM_OP == PIM_OPERATION::JUMP)
    {
        if (LC == 0)
        {
            LC = CRF[PPC].imm1;
            PPC = CRF[PPC].imm0;
        }
        else if (LC > 1)
        {
            PPC = CRF[PPC].imm0;
            LC -= 1;
        }
        else if (LC == 1)
        {
            PPC += 1;
            LC = 0;
        }
        if (DebugMode())
        {
            std::cout << "  PU: JUMP left (" << LC << ")\n";
        }
    }

    // When pointed PIM_INSTRUCTION is EXIT, μkernel is finished
    // Reset PPC and return EXIT_END
    if (CRF[PPC].PIM_OP == PIM_OPERATION::EXIT)
    {
        if (DebugMode())
        {
            std::cout << "  PU: EXIT\n";
        }
        PPC = 0;

        return EXIT_END;
    }

    return 0; // NORMAL_END
}

// Map operand data's offset to computation pointers properly
// AAM mode is controlled in this function
void PimUnit::SetOperandAddr(uint64_t hex_addr)
{
    // set _GRF_A, _GRF_B operand address when AAM mode
    Address addr = AddressMapping(hex_addr);
    if (CRF[PPC].is_aam)
    {
        int ADDR = addr.row * 32 + addr.column;
        int dst_idx = int(ADDR / pow(2, CRF[PPC].dst_idx)) % 8;
        int src0_idx = int(ADDR / pow(2, CRF[PPC].src0_idx)) % 8;
        int src1_idx = int(ADDR / pow(2, CRF[PPC].src1_idx)) % 8;

        int CH = addr.channel;
        int RA = addr.row;
        int BA = addr.bank;
        int CA = addr.column;
        if (DebugMode())
        {
            std::cout << "  PU: " << dst_idx << " " << src0_idx << " " << src1_idx << std::endl;
            std::cout << "  PU: CA=" << CA << ", RA=" << RA << std::endl;
        }
        // int RA = addr.row;
        // int A_idx = CA % 8;
        // int B_idx = CA / 8 + RA % 2 * 4;

        // set dst address (AAM)
        if (CRF[PPC].dst == PIM_OPERAND::GRF_A)
        {
            if (CRF[PPC].is_dst_fix)
            {
                dst = GRF_A_ + CRF[PPC].dst_idx * 16;
            }
            else
            {
                dst = GRF_A_ + dst_idx * 16;
            }
        }
        else if (CRF[PPC].dst == PIM_OPERAND::GRF_B)
        {
            if (CRF[PPC].is_dst_fix)
            {
                dst = GRF_B_ + CRF[PPC].dst_idx * 16;
            }
            else
            {
                dst = GRF_B_ + dst_idx * 16;
            }
        }

        // set src0 address (AAM)
        if (CRF[PPC].src0 == PIM_OPERAND::GRF_A)
        {
            if (CRF[PPC].is_src0_fix)
            {
                src0 = GRF_A_ + CRF[PPC].src0_idx * 16;
            }
            else
            {
                src0 = GRF_A_ + src0_idx * 16;
            }
        }
        else if (CRF[PPC].src0 == PIM_OPERAND::GRF_B)
        {
            if (CRF[PPC].is_src0_fix)
            {
                src0 = GRF_B_ + CRF[PPC].src0_idx * 16;
            }
            else
            {
                src0 = GRF_B_ + src0_idx * 16;
            }
        }
        else if (CRF[PPC].src0 == PIM_OPERAND::SRF_A)
        {
            if (CRF[PPC].is_src0_fix)
            {
                src0 = SRF_A_ + CRF[PPC].src0_idx * 16;
            }
            else
            {
                src0 = SRF_A_ + src0_idx;
            }
        }

        // set src1 address (AAM)
        if (CRF[PPC].src1 == PIM_OPERAND::GRF_A)
        {
            if (CRF[PPC].is_src1_fix)
            {
                src1 = GRF_A_ + CRF[PPC].src1_idx * 16;
            }
            else
            {
                src1 = GRF_A_ + src1_idx * 16;
            }
        }
        else if (CRF[PPC].src1 == PIM_OPERAND::GRF_B)
        {
            if (CRF[PPC].is_src1_fix)
            {
                src1 = GRF_B_ + CRF[PPC].src1_idx * 16;
            }
            else
            {
                src1 = GRF_B_ + src1_idx * 16;
            }
        }
        else if (CRF[PPC].src1 == PIM_OPERAND::SRF_A)
        {
            if (CRF[PPC].is_src1_fix)
            {
                src1 = SRF_A_ + CRF[PPC].src1_idx * 16;
            }
            else
            {
                src1 = SRF_A_ + src1_idx;
            }
        }
        else if (CRF[PPC].src1 == PIM_OPERAND::SRF_M)
        {
            if (CRF[PPC].is_src1_fix)
            {
                src1 = SRF_M_ + CRF[PPC].src1_idx * 16;
            }
            else
            {
                src1 = SRF_M_ + src1_idx;
            }
        }
    }
    else
    { // set _GRF_A, _GRF_B operand address when non-AAM mode
        // set dst address
        if (CRF[PPC].dst == PIM_OPERAND::GRF_A)
            dst = GRF_A_ + CRF[PPC].dst_idx * 16;
        else if (CRF[PPC].dst == PIM_OPERAND::GRF_B)
            dst = GRF_B_ + CRF[PPC].dst_idx * 16;

        // set src0 address
        if (CRF[PPC].src0 == PIM_OPERAND::GRF_A)
            src0 = GRF_A_ + CRF[PPC].src0_idx * 16;
        else if (CRF[PPC].src0 == PIM_OPERAND::GRF_B)
            src0 = GRF_B_ + CRF[PPC].src0_idx * 16;
        else if (CRF[PPC].src0 == PIM_OPERAND::SRF_A)
            src1 = SRF_A_ + CRF[PPC].src1_idx;

        // set src1 address
        // PIM_OP == ADD, MUL, MAC, MAD -> uses src1 for operand
        if (CRF[PPC].pim_op_type == PIM_OP_TYPE::ALU)
        {
            if (CRF[PPC].src1 == PIM_OPERAND::GRF_A)
                src1 = GRF_A_ + CRF[PPC].src1_idx * 16;
            else if (CRF[PPC].src1 == PIM_OPERAND::GRF_B)
                src1 = GRF_B_ + CRF[PPC].src1_idx * 16;
            else if (CRF[PPC].src1 == PIM_OPERAND::SRF_A)
                src1 = SRF_A_ + CRF[PPC].src1_idx;
            else if (CRF[PPC].src1 == PIM_OPERAND::SRF_M)
                src1 = SRF_M_ + CRF[PPC].src1_idx;
        }
    }

    // set BANK, operand address
    // . set dst address
    if (CRF[PPC].dst == PIM_OPERAND::BANK)
        dst = bank_data_;

    // . set src0 address
    if (CRF[PPC].src0 == PIM_OPERAND::BANK)
        src0 = bank_data_;

    // . set src1 address only if PIM_OP_TYPE == ALU
    //   -> uses src1 for operand
    if (CRF[PPC].pim_op_type == PIM_OP_TYPE::ALU)
        if (CRF[PPC].src1 == PIM_OPERAND::BANK)
            src1 = bank_data_;
}

// Execute PIM_INSTRUCTION
void PimUnit::Execute()
{
    if (DebugMode())
    {
        std::cout << "  PU: execute  ";
        PrintPIM_IST(CRF[PPC]);
    }
    switch (CRF[PPC].PIM_OP)
    {
    case PIM_OPERATION::ADD:
        _ADD();
        break;
    case PIM_OPERATION::MUL:
        _MUL();
        break;
    case PIM_OPERATION::MAC:
        _MAC();
        break;
    case PIM_OPERATION::MAD:
        _MAD();
        break;
    case PIM_OPERATION::MOV:
    case PIM_OPERATION::FILL:
        _MOV();
        break;
    default:
        break;
    }
}

void PimUnit::_ADD()
{
    if (CRF[PPC].src1 == PIM_OPERAND::SRF_A)
    {
        for (int i = 0; i < UNITS_PER_WORD; i++)
        {
            dst[i] = src0[i] + src1[0];
        }
    }
    else
    {
        for (int i = 0; i < UNITS_PER_WORD; i++)
        {
            if (DebugMode())
                std::cout << "ADD " << i << "\t" << src0[i] << " " << src1[i] << std::endl;
            dst[i] = src0[i] + src1[i];
        }
    }
}

void PimUnit::_MUL()
{
    for (int i = 0; i < UNITS_PER_WORD; i++)
    {
        dst[i] = src0[i] * src1[i];
    }
}

void PimUnit::_MAC()
{
    if (CRF[PPC].src1 == PIM_OPERAND::SRF_M)
    {
        for (int i = 0; i < UNITS_PER_WORD; i++)
        {
            if (DebugMode())
                std::cout << "MAC " << i << "\t" << src0[i] << "x" << src1[0] << "+" << dst[i] << std::endl;
            dst[i] = src0[i] * src1[0] + dst[i];
        }
    }
    else
    {
        for (int i = 0; i < UNITS_PER_WORD; i++)
        {
            dst[i] = src0[i] * src1[i] + dst[i];
        }
    }
}
void PimUnit::_MAD()
{
    std::cout << "not yet\n";
}

void PimUnit::_MOV()
{
    // std::cout << "(MOV) GRF_B[0]: " << (int)GRF_B_[0] << std::endl;
    for (int i = 0; i < UNITS_PER_WORD; i++)
    {
        dst[i] = src0[i];
    }
}
