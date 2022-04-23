#include "pim_runtime.h"

uint64_t next_addr = 0;
unsigned int burstSize_ = 32;

int ch_pos_ = 0;
int ba_pos_ = ch_pos_ + 4;
int bg_pos_ = ba_pos_ + 2;
int co_pos_ = bg_pos_ + 2;
int ra_pos_ = co_pos_ + 5;
int ro_pos_ = ra_pos_ + 0;
int shift_bits_ = 5;
uint8_t data_temp_[32];
uint64_t ukernel_access_size_ = SIZE_WORD * 8 * NUM_BANK;
uint64_t ukernel_count_per_pim_;

// PIM Preprocessor
bool isSuitableOps(PIM_OP op) {
	std::cout << "  PIM_RUNTIME\t isSuitableOps!\n";
	
	// For now, just return True
	return true;
}

uint8_t* MapMemory(uint8_t *pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t MapMemory!\n";
	uint64_t addr = next_addr;
	next_addr += Ceiling(len * UNIT_SIZE, SIZE_ROW * NUM_BANK);
	return (uint8_t*)(pim_mem+addr);
}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
	std::cout << "  PIM_RUNTIME\t GetMicrokernelCode!\n";
	PIMKernel new_kernel;
	ukernel_count_per_pim_ = Ceiling(op_attrs.len_in * UNIT_SIZE, ukernel_access_size_) / ukernel_access_size_;
	// For now, Make same ukernel without considering op_attrs
	new_kernel.SetMicrokernelCode(op);
	return new_kernel;
}

// PIM Memory Manager
bool AllocMem(uint8_t* pim_mem) {
	std::cout << "  PIM_RUNTIME\t AllocMem!\n";
	
	// For now, I didn't get it
	// So, just return True
	return true;
}

void FreeMem(uint8_t* pim_mem) {
	std::cout << "  PIM_RUNTIME\t FreeMem!\n";
}

size_t ReadMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t ReadMem!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);
	for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
		TryAddTransaction(pim_mem + offset, data + offset, false); 
}

size_t WriteMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t WriteMem!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);

	for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
		TryAddTransaction(pim_mem + offset, data + offset, true); 
}

// PIM Kernel Executor
bool ExecuteKernel(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
	std::cout << "  PIM_RUNTIME\t ExecuteKernel!\n";

	switch(pim_kernel.pim_op) {
		case (PIM_OP::ADD):
		case (PIM_OP::MUL):
			ElementWiseExecute(pim_mem, pim_x, pim_y, pim_z, pim_kernel);
			break;
		case (PIM_OP::GEMV):
			GemvExecute(pim_mem, pim_x, pim_y, pim_z, pim_kernel);
			break;
		case (PIM_OP::BN):
			BnExecute(pim_mem, pim_x, pim_y, pim_z, pim_kernel);
			break;
		case (PIM_OP::LSTM):
			LstmExecute(pim_mem, pim_x, pim_y, pim_z, pim_kernel);
			break;
	}
	return 1;
}

void ElementWiseExecute(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
    // ro : row index in bank
    // co_o(column_out) : column index counting by 8 words in bank
    // co_i(column_in) : column index counting by word in co_o(column_out)
	std::cout << "execute elementwise\n";
	
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            // Check that all data operations have been completed
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_)
                break;

            // Mode transition: AB -> AB-PIM
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }

            // Execute ukernel 0-1
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, false);
                }
            }

            /* >> mmm for clflush optim. */
            // Execute ukernel 2-3
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co);
                    TryAddTransaction(pim_y + hex_addr, data_temp_, false);
                }
            }
            // mmm << for clflush optim.

            // Execute ukernel 4-5
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co);
                    TryAddTransaction(pim_z + hex_addr, data_temp_, true);
                }
            }

            // Execute ukernel 6-7
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, false);
                }
            }

            /* >> mmm for clflush optim. */
            // Execute ukernel 8-9
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co);
                    TryAddTransaction(pim_y + hex_addr, data_temp_, false);
                }
            }
            // mmm << for clflush optim.

            // Execute ukernel 10-11 + AB-PIM -> AB
            // AB-PIM -> AB occurs automatically at the end of the kernel(EXIT)
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co);
                    TryAddTransaction(pim_z + hex_addr, data_temp_, true);
                }
            }
        }
    }	
}

void BnExecute(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
	std::cout << "execute bn\n";
    // ro : row index in bank
    // co_o(column_out) : column index counting by 8 words in bank
    // co_i(column_in) : column index counting by word in co_o(column_out)
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            // Check that all data operations have been completed
            if (ro * NUM_WORD_PER_ROW / 8 + co_o > ukernel_count_per_pim_)
                break;

            // Mode transition: AB -> AB-PIM
            *data_temp_ |= 1;
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }
            
            // Execute ukernel 0-1
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, false);
                }
            }
            
            // Execute ukernel 2-3
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co_i);
                    TryAddTransaction(pim_y + hex_addr, data_temp_, false);
                }
            }
            
            // Execute ukernel 4-5
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co_i);
					//std::cout << co << " " << co_o << "\n";
                    TryAddTransaction(pim_z + hex_addr, data_temp_, false);
                }
            }

            // Execute ukernel 6-7
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, EVEN_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, true);
                }
            }

            // Execute ukernel 8-9
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, false);
                }
            }
        
            // Execute ukernel 10-11
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co_i);
                    TryAddTransaction(pim_y + hex_addr, data_temp_, false);
                }
            }

            // Execute ukernel 12-13
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co_i);
                    TryAddTransaction(pim_z + hex_addr, data_temp_, false);
                }
            }

            // Execute ukernel 14-15 + AB-PIM -> AB
            // AB-PIM -> AB occurs automatically at the end of the kernel(EXIT)
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, ODD_BANK, ro, co);
                    TryAddTransaction(pim_x + hex_addr, data_temp_, true);
                }
            }
        }
    }
}

void GemvExecute(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
	std::cout << "execute gemv\n";
    GemvExecuteBank(EVEN_BANK, pim_mem, pim_x, pim_y, pim_z, pim_kernel);
    GemvExecuteBank(ODD_BANK, pim_mem, pim_x, pim_y, pim_z, pim_kernel);
}

void GemvExecuteBank(bool bank, uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
    std::cout << "execute gemv bank\n";
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, co);
            TryAddTransaction(pim_mem + hex_addr, (uint8_t*)&pim_kernel.ukernel[co*8], true);
        }
    }

    // Execute for EVEN_BANK or ODD_BANK
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            // SRF_M modify
            std::memcpy(data_temp_ + 16,
                        ((uint16_t*)pim_x) + (ro * NUM_WORD_PER_ROW + co_o * 8),
                        16);

            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, bank, MAP_SRF, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }

            // if last gemv ukernel to execute, add new gemv ukernel (= ukernel_gemv_last)
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_-1) {
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    for (int co = 0; co < 4; co++) {
                        uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, co);
                        TryAddTransaction(pim_mem + hex_addr, (uint8_t*)&pim_kernel.ukernel_extra[co*8], true);
                    }
                }
            }

            // Mode transition: AB -> AB-PIM
            #ifdef debug_mode
            std::cout << "\nHOST:\t[2] AB -> PIM \n";
            #endif
            *data_temp_ |= 1;
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }

            // Execute ukernel 0-1 + AB-PIM -> AB
            #ifdef debug_mode
            std::cout << "\nHOST:\tExecute μkernel 0-1 + [3] PIM -> AB \n";
            #endif
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, bank, ro, co);
                    TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
                }
            }

            // for the last gemv ukernel, move result to bank
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_-1) {
                for (int uker = 0; uker < 1; uker++) {
                    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                        uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
                        TryAddTransaction(pim_z + hex_addr, data_temp_, true);
                    }
                }
                break;
            }
        }
    }

    // reset GRF_B
    uint8_t* zero = (uint8_t*)malloc(WORD_SIZE);
    for (int i=0; i< WORD_SIZE; i++) zero[i] = 0;
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 8; co < 16; co++) {
            uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, co);
            TryAddTransaction(pim_mem + hex_addr, zero, true);
        }
    }
}

void LstmExecute(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel) {
	std::cout << "execute lstm\n";
    LstmExecuteBank(EVEN_BANK, pim_mem, pim_x, pim_y, pim_z, pim_kernel);
    LstmExecuteBank(ODD_BANK, pim_mem, pim_x, pim_y, pim_z, pim_kernel);
}

void LstmExecuteBank(bool bank, uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIMKernel pim_kernel){
    std::cout << "execute lstm bank\n";
	/*
    // Program lstm μkernel
    #ifdef debug_mode
    std::cout << "HOST:\tProgram lstm μkernel \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, co);
            TryAddTransaction(pim_mem + hex_addr, (uint8_t*)&pim_kernel.ukernel[co*8], true);
        }
    }

    // Execute for EVEN_BANK or ODD_BANK
    for (int ro = 0; ro * NUM_WORD_PER_ROW / 8 < ukernel_count_per_pim_; ro++) {
        for (int co_o = 0; co_o < NUM_WORD_PER_ROW / 8; co_o++) {
            #ifdef debug_mode
            std::cout << "\nHOST:\tSet Srf\n";
            #endif
            // SRF_M modify
            std::memcpy(data_temp_ + 16,
                        ((uint16_t*)h_) + (ro * NUM_WORD_PER_ROW + co_o * 8),
                        16);
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, bank, MAP_SRF, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }

            // Mode transition: AB -> AB-PIM
            #ifdef debug_mode
            std::cout << "\nHOST:\t[2] AB -> PIM \n";
            #endif
            *data_temp_ |= 1;
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
                TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
            }

            // Execute ukernel 0-1 + AB-PIM -> AB
            #ifdef debug_mode
            std::cout << "\nHOST:\tExecute μkernel 0-1 + [3] PIM -> AB \n";
            #endif
            for (int co_i = 0; co_i < 8; co_i++) {
                uint64_t co = co_o * 8 + co_i;
                for (int ch = 0; ch < NUM_CHANNEL; ch++) {
                    uint64_t hex_addr = GetAddress(ch, 0, 0, bank, ro, co);
                    TryAddTransaction(addr_Wh_ + hex_addr, data_temp_, false);
                }
            }

            // for the last gemv ukernel, move result to bank
            if (ro * NUM_WORD_PER_ROW / 8 + co_o >= ukernel_count_per_pim_)
                break;
        }
    }

    // Program wr_result ukernel
    #ifdef debug_mode
    std::cout << "\nHOST:\tProgram wr_result μkernel \n";
    #endif
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 0; co < 4; co++) {
            uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, co);
            TryAddTransaction(pim_mem + hex_addr, (uint8_t*)&pim_kernel.ukernel_extra[co*8], true);
        }
    }

    // Mode transition: AB -> AB-PIM
    #ifdef debug_mode
    std::cout << "\nHOST:\t[4] AB -> PIM \n";
    #endif
    *data_temp_ |= 1;
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
        TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
    }

    // Execute ukernel 0~2 + AB-PIM -> AB
    #ifdef debug_mode
    std::cout << "\nHOST:\tExecute μkernel 0-2 + [5] PIM -> AB \n";
    #endif
        
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
        TryAddTransaction(addr_b_ + hex_addr, data_temp_, false);
    }
        
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
        TryAddTransaction(addr_x_ + hex_addr, data_temp_, false);
    }

    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
        TryAddTransaction(addr_y_ + hex_addr, data_temp_, true);
    }

    // reset GRF_B
    #ifdef debug_mode
    std::cout << "\nHOST:\tReset GRF_B\n";
    #endif
    uint8_t* zero = (uint8_t*)malloc(WORD_SIZE);
    for (int i=0; i< WORD_SIZE; i++) zero[i] = 0;
    for (int ch = 0; ch < NUM_CHANNEL; ch++) {
        for (int co = 8; co < 16; co++) {
            uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, co);
            TryAddTransaction(pim_mem + hex_addr, zero, true);
        }
    }
	*/
}


// Some tool
uint64_t Ceiling(uint64_t num, uint64_t stride) {
	return ((num + stride - 1) / stride) * stride;
}

void TryAddTransaction(uint8_t* pim_mem, uint8_t* data, bool is_write) {
	if (is_write)
		std::memcpy(pim_mem, data, burstSize_);
	else
		std::memcpy(data, pim_mem, burstSize_);
}

uint64_t GetAddress(int channel, int rank, int bankgroup, int bank, int row, int column) {
	uint64_t hex_addr = 0;
	hex_addr += ((uint64_t)channel) << ch_pos_;
	hex_addr += ((uint64_t)rank) << ra_pos_;
	hex_addr += ((uint64_t)bankgroup) << bg_pos_;
	hex_addr += ((uint64_t)bank) << ba_pos_;
	hex_addr += ((uint64_t)row) << ro_pos_;
	hex_addr += ((uint64_t)column) << co_pos_;
	return hex_addr << shift_bits_;
}
