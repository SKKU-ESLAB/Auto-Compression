#include "transaction_generator.h"
#include <x86intrin.h>

using half_float::half;

Address TransactionGenerator::AddressMapping(uint64_t hex_addr) const {
    hex_addr >>= shift_bits_;
    int channel = (hex_addr >> ch_pos_) & ch_mask_;
    int rank = (hex_addr >> ra_pos_) & ra_mask_;
    int bg = (hex_addr >> bg_pos_) & bg_mask_;
    int ba = (hex_addr >> ba_pos_) & ba_mask_;
    int ro = (hex_addr >> ro_pos_) & ro_mask_;
    int co = (hex_addr >> co_pos_) & co_mask_;
    return Address(channel, rank, bg, ba, ro, co);
}

// Map 64-bit hex_address into structured address
uint64_t TransactionGenerator::ReverseAddressMapping(Address& addr) {
    uint64_t hex_addr = 0;
    hex_addr += ((uint64_t)addr.channel) << ch_pos_;
    hex_addr += ((uint64_t)addr.rank) << ra_pos_;
    hex_addr += ((uint64_t)addr.bankgroup) << bg_pos_;
    hex_addr += ((uint64_t)addr.bank) << ba_pos_;
    hex_addr += ((uint64_t)addr.row) << ro_pos_;
    hex_addr += ((uint64_t)addr.column) << co_pos_;
    return hex_addr << shift_bits_;
}

// Returns the minimum multiple of stride that is higher than num
uint64_t TransactionGenerator::Ceiling(uint64_t num, uint64_t stride) {
    return ((num + stride - 1) / stride) * stride;
}

// Send transaction to memory_system (DRAMsim3 + PIM Functional Simulator)
//  hex_addr : address to RD/WR from physical memory or change bank mode
//  is_write : denotes to Read or Write
//  *DataPtr : buffer used for both RD/WR transaction (read common.h)
void TransactionGenerator::TryAddTransaction(uint64_t hex_addr, bool is_write,
                                             uint8_t *DataPtr) {
    // Send transaction to memory_system
    if (is_write) {
        //uint8_t *new_data = (uint8_t *) malloc(burstSize_);
        //std::memcpy(new_data, DataPtr, burstSize_);
        //memory_system_.AddTransaction(hex_addr, is_write, new_data);
        //std::cout << "w " << hex_addr << " " << tmp << std::endl;
        //tmp++;
        std::memcpy(&pmemAddr_[hex_addr], DataPtr, burstSize_);
    } else {
        //memory_system_.AddTransaction(hex_addr, is_write, DataPtr);
        //std::cout << "r " << hex_addr << " " << tmp << std::endl;
        //tmp++;
        std::memcpy(DataPtr, &pmemAddr_[hex_addr], burstSize_);
    }
}

// Prevent turning out of order between transaction parts
//  Change memory's threshold and wait until all pending transactions are
//  executed
void TransactionGenerator::Barrier() {
    _mm_mfence();
    return;
#if 0
    memory_system_.SetWriteBufferThreshold(0);
    while (memory_system_.IsPendingTransaction()) {
        memory_system_.ClockTick();
        clk_++;
    }
    memory_system_.SetWriteBufferThreshold(-1);
#endif
}

// Initialize variables and ukernel
void AddTransactionGenerator::Initialize() {
    // base address of operands
    addr_x_ = 0;
    addr_y_ = Ceiling(n_ * UNIT_SIZE, SIZE_ROW * NUM_BANK);
    //addr_z_ = addr_y_ + Ceiling(n_ * UNIT_SIZE, SIZE_ROW * NUM_BANK);
    addr_z_ = addr_y_; // >> mmm << for clflush optim.

    // total access size of one operand in one ukernel cycle
    ukernel_access_size_ = SIZE_WORD * 8 * NUM_BANK;

    // number of total ukernel cycles to run the whole computation
    ukernel_count_per_pim_ = Ceiling(n_ * UNIT_SIZE, ukernel_access_size_)
                                     / ukernel_access_size_;

    // Define ukernel
    ukernel_ = (uint32_t *) malloc(sizeof(uint32_t) * 32);
    ukernel_[0]  = 0b01000010000000001000000000000000; // MOV(AAM)  GRF_A  BANK
    ukernel_[1]  = 0b00010000000001000000100000000111; // JUMP      -1     7
    ukernel_[2]  = 0b10000010000010001000000000000000; // ADD(AAM)  GRF_A  BANK  GRF_A
    ukernel_[3]  = 0b00010000000001000000100000000111; // JUMP      -1     7
    //ukernel_[4]= 0b00000000000000000000000000001000; // NOP       8    // >> mmm << for clflush optim.
    ukernel_[4]  = 0b01000000010000001000000000000000; // MOV(AAM)  BANK   GRF_A
    ukernel_[5]  = 0b00010000000001000000100000000111; // JUMP      -1     7
    ukernel_[6]  = 0b01000010000000001000000000000000; // MOV(AAM)  GRF_A  BANK
    ukernel_[7]  = 0b00010000000001000000100000000111; // JUMP      -1     7
    ukernel_[8]  = 0b10000010000010001000000000000000; // ADD(AAM)  GRF_A  BANK  GRF_A
    ukernel_[9]  = 0b00010000000001000000100000000111; // JUMP      -1     7
    //ukernel_[10]=0b00000000000000000000000000001000; // NOP       8    // >> mmm << for clflush optim.
    ukernel_[10] = 0b01000000010000001000000000000000; // MOV(AAM)  BANK   GRF_A
    ukernel_[11] = 0b00010000000001000000100000000111; // JUMP      -1     7
    ukernel_[12] = 0b00100000000000000000000000000000; // EXIT
}

void clflush(uint8_t *mem, size_t size) {
    for (size_t i = 0; i < size; i += 32)
        _mm_clflush(&mem[i]);
}

// Write operand data and μkernel to physical memory and PIM registers
void AddTransactionGenerator::SetData() {
    // strided size of one operand with one computation part(minimum)
    uint64_t strided_size = Ceiling(n_ * UNIT_SIZE, SIZE_WORD * NUM_BANK);

    #ifdef debug_mode
    std::cout << "HOST:\tSet input data\n";
    #endif
    // Write input data x to physical memory
    for (int offset = 0; offset < strided_size ; offset += SIZE_WORD)
        TryAddTransaction(addr_x_ + offset, true, x_ + offset);
    clflush(&pmemAddr_[addr_x_], strided_size);

    // Write input data y to physical memory
    for (int offset = 0; offset < strided_size ; offset += SIZE_WORD)
        TryAddTransaction(addr_y_ + offset, true, y_ + offset);
    clflush(&pmemAddr_[addr_y_], strided_size);
    Barrier();

    time_t start, end;
    double result;

    start = clock();
    // Mode transition: SB -> AB
    #ifdef debug_mode
    std::cout << "\nHOST:\t[1] SB -> AB \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_ABMR, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(hex_addr, false, data_temp_);
    }
    Barrier();

    // Program μkernel into CRF register
    #ifdef debug_mode
    std::cout << "\nHOST:\tProgram μkernel \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            Address addr(ch, 0, 0, 0, MAP_CRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            TryAddTransaction(hex_addr, true, (uint8_t*)&ukernel_[co*8]);
            clflush(&pmemAddr_[hex_addr], 32);
        }
    }
    Barrier();
   
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken for SB->AB & SetCRF: " << (result/CLOCKS_PER_SEC) << "sec \n";
}

// Execute PIM computation
void AddTransactionGenerator::Execute() {
    // ro : row index in bank
    // co_o(column_out) : column index counting by 8 words in bank
    // co_i(column_in) : column index counting by word in co_o(column_out)
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            // Check that all data operations have been completed
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_)
                break;

            // Mode transition: AB -> AB-PIM
            #ifdef debug_mode
            std::cout << "HOST:\t[2] AB -> PIM \n";
            #endif
            *data_temp_ |= 1;
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                Address addr(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                uint64_t hex_addr = ReverseAddressMapping(addr);
                TryAddTransaction(hex_addr, true, data_temp_);
                clflush(&pmemAddr_[hex_addr], 32);
            }
            Barrier();

            #ifdef debug_mode
            std::cout << "\nHOST:\tExecute μkernel 0-9\n";
            #endif
            // Execute ukernel 0-1
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, EVEN_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_x_ + hex_addr, false, data_temp_);
                }
            }
            Barrier();

            /* >> mmm for clflush optim.
            // Execute ukernel 2-3
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, EVEN_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_y_ + hex_addr, false, data_temp_);
                }
            }
            Barrier();
            */ // mmm << for clflush optim.

            // Execute ukernel 4-5
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, EVEN_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_z_ + hex_addr, true, data_temp_);
                }
            }

            // Flush!!
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, EVEN_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    clflush(&pmemAddr_[addr_z_+hex_addr], 32);
                }
            }
            Barrier();

            // Execute ukernel 6-7
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, ODD_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_x_ + hex_addr, false, data_temp_);
                }
            }
            Barrier();

            /* >> mmm for clflush optim.
            // Execute ukernel 8-9
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, ODD_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_y_ + hex_addr, false, data_temp_);
                }
            }
            Barrier();
            */ // mmm << for clflush optim.

            // Execute ukernel 10-11 + AB-PIM -> AB
            // AB-PIM -> AB occurs automatically at the end of the kernel(EXIT)
            #ifdef debug_mode
            std::cout << "\nHOST:\tExecute μkernel 10-11 + [3] PIM -> AB \n";
            #endif
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, ODD_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(addr_z_ + hex_addr, true, data_temp_);
                }
            }
            // Flush!!
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, ODD_BANK, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    clflush(&pmemAddr_[addr_z_+hex_addr], 32);
                }
            }
            Barrier();
        }
    }
}

// Read PIM computation result from physical memory
void AddTransactionGenerator::GetResult() {
    // Mode transition: AB -> SB
    #ifdef debug_mode
    std::cout << "HOST:\t[4] AB -> SB \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_SBMR, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(hex_addr, false, data_temp_);
    }
    Barrier();

    uint64_t strided_size = Ceiling(n_ * UNIT_SIZE, SIZE_WORD * NUM_BANK);
    // Read output data z
    #ifdef debug_mode
    std::cout << "\nHOST:\tRead output data z\n";
    #endif
    for (int offset = 0; offset < strided_size ; offset += SIZE_WORD)
        TryAddTransaction(addr_z_ + offset, false, z_ + offset);
    Barrier();
}

// Calculate error between the result of PIM computation and actual answer
void AddTransactionGenerator::CheckResult() {
    half h_err(0);
    half h_ans(0);
    half h_zzz(0);
    uint8_t *answer = (uint8_t *) malloc(sizeof(uint16_t) * n_);

    // Calculate actual answer of GEMV
    for (int i=0; i< n_; i++) {
        half h_x(*reinterpret_cast<half*>(&((uint16_t*)x_)[i]));
        half h_y(*reinterpret_cast<half*>(&((uint16_t*)y_)[i]));
        half h_answer = h_x + h_y;
        ((uint16_t*)answer)[i] = *reinterpret_cast<uint16_t*>(&h_answer);
    }

    // Calculate error
    for (int i=0; i< n_; i++) {
        half h_answer(*reinterpret_cast<half*>(&((uint16_t*)answer)[i]));
        half h_z(*reinterpret_cast<half*>(&((uint16_t*)z_)[i]));
        h_ans += fabs(h_answer);
        h_zzz += fabs(h_z);
        h_err += fabs(h_answer - h_z);  // fabs stands for float absolute value
    }
    std::cout << "answer : " << h_ans << std::endl;
    std::cout << "pim result : " << h_zzz << std::endl;
    std::cout << "ERROR : " << h_err << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

// Initialize variables and ukernel
void GemvTransactionGenerator::Initialize() {
    // TODO(bepo): currently only support m=4096

    addr_A_ = 0;
    addr_y_ = Ceiling(m_ * n_ * UNIT_SIZE, SIZE_ROW * NUM_BANK);

    ukernel_access_size_ = SIZE_WORD * 8 * NUM_BANK;
    ukernel_count_per_pim_ = Ceiling(m_ * n_ * UNIT_SIZE, ukernel_access_size_)
                                     / ukernel_access_size_;

    // Define ukernel for gemv
    ukernel_gemv_ = (uint32_t *) malloc(sizeof(uint32_t) * 32);
    for (int i=0; i< 32; i++)
        ukernel_gemv_[i] = 0b00000000000000000000000000000000; // initialize

    ukernel_gemv_[0] = 0b10100100001000001000000000000000; // MAC(AAM)  GRF_B  BANK  SRF_M
    ukernel_gemv_[1] = 0b00010000000001000000100000000111; // JUMP      -1     7
    ukernel_gemv_[2] = 0b00100000000000000000000000000000; // EXIT

    // Define ukernel for reducing output data from ukernel_gemv + write to
    // physical memory
    ukernel_reduce_ = (uint32_t *) malloc(sizeof(uint32_t) * 32);
    for (int i=0; i< 32; i++)
		ukernel_reduce_[i] = 0b00000000000000000000000000000000; // initialize

	ukernel_reduce_[0] = 0b10000100100100000000000000000001; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[1]
    ukernel_reduce_[1] = 0b10000100100100000000000000000010; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[2]
    ukernel_reduce_[2] = 0b10000100100100000000000000000011; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[3]
    ukernel_reduce_[3] = 0b10000100100100000000000000000100; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[4]
    ukernel_reduce_[4] = 0b10000100100100000000000000000101; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[5]
    ukernel_reduce_[5] = 0b10000100100100000000000000000110; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[6]
    ukernel_reduce_[6] = 0b10000100100100000000000000000111; // ADD  GRF_B[0]  GRF_B[0]  GRF_B[7]
    ukernel_reduce_[7] = 0b00000000000000000000000000000001; // NOP  1
    ukernel_reduce_[8] = 0b01000000100000000000000000000000; // MOV  BANK      GRF_B[0]
    ukernel_reduce_[9] = 0b00100000000000000000000000000000; // EXIT
}

// Write operand data and μkernel to physical memory and PIM registers
void GemvTransactionGenerator::SetData() {
    uint64_t strided_size = Ceiling(m_ * n_ * UNIT_SIZE, SIZE_WORD * NUM_BANK);

    // Transpose input data A
    A_T_ = (uint8_t *) malloc(sizeof(uint16_t) * m_ * n_);
    for (int m = 0; m < m_; m++) {
        for (int n = 0; n < n_; n++) {
            ((uint16_t*)A_T_)[n * m_ + m] = ((uint16_t*)A_)[m * n_ + n];
        }
    }

    #ifdef debug_mode
    std::cout << "HOST:\tSet input data\n";
    #endif
    // Write input data A
    for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
        TryAddTransaction(addr_A_ + offset, true, A_T_ + offset);
    clflush(&pmemAddr_[addr_A_], strided_size);
    Barrier();

    time_t start, end;
    double result;
    
    start = clock();
    // Mode transition: SB -> AB
    #ifdef debug_mode
    std::cout << "\nHOST:\t[1] SB -> AB \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_ABMR, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(hex_addr, false, data_temp_);
    }
    Barrier();

    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken for SB->AB: " << (result/CLOCKS_PER_SEC) << "sec \n\n";
}

// Execute PIM computation
void GemvTransactionGenerator::Execute() {
    ExecuteBank(EVEN_BANK);
    ExecuteBank(ODD_BANK);
}

// Execute PIM computation of EVEN_BANK or ODD_BANK
void GemvTransactionGenerator::ExecuteBank(int bank) {
    // Program gemv μkernel
    #ifdef debug_mode
    std::cout << "HOST:\tProgram gemv μkernel \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            Address addr(ch, 0, 0, 0, MAP_CRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            TryAddTransaction(hex_addr, true, (uint8_t*)&ukernel_gemv_[co*8]);
        }
    }
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            Address addr(ch, 0, 0, 0, MAP_CRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            clflush(&pmemAddr_[hex_addr], 32);
        }
    }
    Barrier();

    // Execute for EVEN_BANK or ODD_BANK
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            std::memcpy(data_temp_ + 16,
                        ((uint16_t*)x_) + (ro * NUM_WORD_PER_ROW + co_o * 8),
                        16);

            #ifdef debug_mode
            std::cout << "\nHOST:\tSet Srf\n";
            #endif
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                Address addr(ch, 0, 0, bank, MAP_SRF, 0);
                uint64_t hex_addr = ReverseAddressMapping(addr);
                TryAddTransaction(hex_addr, true, data_temp_);
            }
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                Address addr(ch, 0, 0, bank, MAP_SRF, 0);
                uint64_t hex_addr = ReverseAddressMapping(addr);
                clflush(&pmemAddr_[hex_addr], 32);
            }
            Barrier();

            // Mode transition: AB -> AB-PIM
            #ifdef debug_mode
            std::cout << "\nHOST:\t[2] AB -> PIM \n";
            #endif
            *data_temp_ |= 1;
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                Address addr(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                uint64_t hex_addr = ReverseAddressMapping(addr);
                TryAddTransaction(hex_addr, true, data_temp_);
            }
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                Address addr(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                uint64_t hex_addr = ReverseAddressMapping(addr);
                clflush(&pmemAddr_[hex_addr], 32);
            }
            Barrier();

            // Execute ukernel 0-1 + AB-PIM -> AB
            #ifdef debug_mode
            std::cout << "\nHOST:\tExecute μkernel 0-1 + [3] PIM -> AB \n";
            #endif
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    Address addr(ch, 0, 0, bank, ro, co);
                    uint64_t hex_addr = ReverseAddressMapping(addr);
                    TryAddTransaction(hex_addr, false, data_temp_);
                }
            }
            Barrier();

            // Check that all data operations have been completed
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_)
                break;
        }
    }

    // Program reduce ukernel
    #ifdef debug_mode
    std::cout << "\nHOST:\tProgram reduce μkernel \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            Address addr(ch, 0, 0, 0, MAP_CRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            TryAddTransaction(hex_addr, true, (uint8_t*)&ukernel_reduce_[co*8]);
        }
    }
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            Address addr(ch, 0, 0, 0, MAP_CRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            clflush(&pmemAddr_[hex_addr], 32);
        }
    }
    Barrier();

    // Mode transition: AB -> AB-PIM
    #ifdef debug_mode
    std::cout << "\nHOST:\t[4] AB -> PIM \n";
    #endif
    *data_temp_ |= 1;
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(hex_addr, true, data_temp_);
    }
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        clflush(&pmemAddr_[hex_addr], 32);
    }
    Barrier();

    // Execute ukernel 0~6
    #ifdef debug_mode
    std::cout << "\nHOST:\tExecute μkernel 0-6\n";
    #endif
    for (int uker = 0; uker < 7; uker++) {
        for (int ch = 0; ch < NUM_CHANNEL; ch++) {
            Address addr(ch, 0, 0, bank, 0, 0);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            TryAddTransaction(addr_y_ + hex_addr, false, data_temp_);
        }
        for (int ch = 0; ch < NUM_CHANNEL; ch++) {
            Address addr(ch, 0, 0, bank, 0, 0);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            clflush(&pmemAddr_[addr_y_ + hex_addr], 32);
        }
        Barrier();
    }

    // Execute ukernel 7 + AB-PIM -> AB
    #ifdef debug_mode
    std::cout << "\nHOST:\tExecute μkernel 7 + [5] PIM -> AB \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, bank, 0, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(addr_y_ + hex_addr, true, data_temp_);
    }
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, bank, 0, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        clflush(&pmemAddr_[addr_y_ + hex_addr], 32);
    }
    Barrier();


    // reset GRF_B
    #ifdef debug_mode
    std::cout << "\nHOST:\tReset GRF_B\n";
    #endif
    uint8_t* zero = (uint8_t*)malloc(WORD_SIZE);
    for (int i=0; i< WORD_SIZE; i++) zero[i] = 0;
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 8; co < 16; co++) {
            Address addr(ch, 0, 0, 0, MAP_GRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            TryAddTransaction(hex_addr, true, zero);
        }
    }
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 8; co < 16; co++) {
            Address addr(ch, 0, 0, 0, MAP_GRF, co);
            uint64_t hex_addr = ReverseAddressMapping(addr);
            clflush(&pmemAddr_[hex_addr], 32);
        }
    }
    Barrier();
}

// Read PIM computation result from physical memory
void GemvTransactionGenerator::GetResult() {
    // Mode transition: AB -> SB
    #ifdef debug_mode
    std::cout << "HOST:\t[4] AB -> SB \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        Address addr(ch, 0, 0, 0, MAP_SBMR, 0);
        uint64_t hex_addr = ReverseAddressMapping(addr);
        TryAddTransaction(hex_addr, false, data_temp_);
    }
    Barrier();

    uint64_t strided_size = Ceiling(m_ * UNIT_SIZE, SIZE_WORD * NUM_BANK);
    // Read output data z
    #ifdef debug_mode
    std::cout << "\nHOST:\tRead output data z\n";
    #endif
    for (int offset = 0; offset < strided_size ; offset += SIZE_WORD)
        TryAddTransaction(addr_y_ + offset, false, y_ + offset);
    Barrier();
}

// Calculate error between the result of PIM computation and actual answer
void GemvTransactionGenerator::CheckResult() {
    half h_err(0);
    half h_ans(0);
    half h_yyy(0);
    uint8_t *answer = (uint8_t *) malloc(sizeof(uint16_t) * m_);

    // Calculate actual answer of GEMV
    for (int m=0; m<m_; m++) {
        half h_answer(0);
        for(int n_grf=0; n_grf<8; n_grf++) {
            half h_answer_one_grf(0);
            for (int no=0; no*64+n_grf*8<n_; no++) {
                for (int ni=0; ni<8; ni++) {
                    int n = no * 64 + n_grf * 8 + ni;
                    half h_A(*reinterpret_cast<half*>(&((uint16_t*)A_)[m*n_+n]));
                    half h_x(*reinterpret_cast<half*>(&((uint16_t*)x_)[n]));
                    h_answer_one_grf = fma(h_A, h_x, h_answer_one_grf);
                }
            }
            h_answer = h_answer + h_answer_one_grf;
        }
        ((uint16_t*)answer)[m] = *reinterpret_cast<uint16_t*>(&h_answer);
    }

    // Calculate error
    for (int m=0; m< m_; m++) {
        half h_answer(*reinterpret_cast<half*>(&((uint16_t*)answer)[m]));
        half h_y(*reinterpret_cast<half*>(&((uint16_t*)y_)[m]));
        h_ans += fabs(h_answer);
        h_yyy += fabs(h_y);
        h_err += fabs(h_answer - h_y);  // fabs stands for float absolute value
    }
    std::cout << "answer : " << h_ans << std::endl;
    std::cout << "pim result : " << h_yyy << std::endl;
    std::cout << "ERROR : " << h_err << std::endl;
}
