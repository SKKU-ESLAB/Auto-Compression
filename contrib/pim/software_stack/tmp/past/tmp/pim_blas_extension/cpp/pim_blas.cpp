#include <torch/extension.h>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>

///////////////////////////////////////////////////////////////
// >> kkm << I don't know how to extend transaction_generator.h
// >> kkm << so just paste it
///////////////////////////////////////////////////////////////

// >> pim_config.h << // 
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

// >> PIM-DD.h <<
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


// >> transaction_generator.h << // 
#define EVEN_BANK 0
#define ODD_BANK  1

#define NUM_WORD_PER_ROW     32
#define NUM_CHANNEL          16
#define NUM_BANK_PER_CHANNEL 16
#define NUM_BANK             (NUM_BANK_PER_CHANNEL * NUM_CHANNEL)
#define SIZE_WORD            32
#define SIZE_ROW             (SIZE_WORD * NUM_WORD_PER_ROW)
#define NUM_THREAD_PER_CHANNEL 16

#define MAP_SBMR             0x3fff
#define MAP_ABMR             0x3ffe
#define MAP_PIM_OP_MODE      0x3ffd
#define MAP_CRF              0x3ffc
#define MAP_GRF              0x3ffb
#define MAP_SRF              0x3ffa

#define C_NORMAL "\033[0m"
#define C_RED    "\033[031m"
#define C_GREEN  "\033[032m"
#define C_YELLOW "\033[033m"
#define C_BLUE   "\033[034m"


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

struct ThrInput {
    ThrInput(uint64_t hex_addr, bool is_write, uint8_t *DataPtr)
        : hex_addr(hex_addr),
          is_write(is_write),
          DataPtr(DataPtr) {}
    uint64_t hex_addr;
    bool is_write;
    uint8_t* DataPtr;
};

class TransactionGenerator {
 public:
    TransactionGenerator() {
        fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
        if (fd_ < 0)
        {
            printf("open fail %s\n", strerror(errno));
            exit(1);
        }
        printf("fd %d\n", fd_);

        pmemAddr_ = (uint8_t *) mmap(NULL, LEN_PIM,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED,
                                     fd_, 0);
        if (pmemAddr_ == (uint8_t*) MAP_FAILED)
            perror("mmap");

        printf("finish mmap at %llx\n", pmemAddr_);

        burstSize_ = 32;

        ch_pos_ = 0;
        ba_pos_ = ch_pos_ + 4;
        bg_pos_ = ba_pos_ + 2;
        co_pos_ = bg_pos_ + 2;
        ra_pos_ = co_pos_ + 5;
        ro_pos_ = ra_pos_ + 0;
        shift_bits_ = 5;

        ch_mask_ = (1 << 4) - 1;
        ra_mask_ = (1 << 0) - 1;
        bg_mask_ = (1 << 2) - 1;
        ba_mask_ = (1 << 2) - 1;
        ro_mask_ = (1 << 14) - 1;
        co_mask_ = (1 << 5) - 1;
    }
    ~TransactionGenerator() {
        munmap(pmemAddr_, LEN_PIM);
        close(fd_);
    }
    // virtual void ClockTick() = 0;
    virtual void Initialize() = 0;
    virtual void SetData() = 0;
    virtual void Execute() = 0;
    virtual void GetResult() = 0;
	virtual void CheckResult() = 0;

    Address AddressMapping(uint64_t hex_addr) const;
    uint64_t ReverseAddressMapping(Address& addr);
    int GetChannel(uint64_t hex_addr);
    uint64_t Ceiling(uint64_t num, uint64_t stride);
    void TryAddTransaction(uint64_t hex_addr, bool is_write, uint8_t *DataPtr);
    void Barrier();

 protected:
    int fd_;
    int tmp;
    uint8_t *pmemAddr_;
    unsigned int burstSize_;

    uint8_t data_temp_[32];
    uint8_t write_data_[32];

    int shift_bits_;
    int ch_pos_, ra_pos_, bg_pos_, ba_pos_, ro_pos_, co_pos_;
    uint64_t ch_mask_, ra_mask_, bg_mask_, ba_mask_, ro_mask_, co_mask_;
};

class AddTransactionGenerator : public TransactionGenerator {
 public:
    AddTransactionGenerator(uint64_t n,
                            uint8_t *x,
                            uint8_t *y,
                            uint8_t *z)
        : n_(n), x_(x), y_(y), z_(z) {}
    void Initialize() override;
    void SetData() override;
    void Execute() override;
    void GetResult() override;
    void CheckResult() override;

 private:
    uint8_t *x_, *y_, *z_;
    uint64_t n_;
    uint64_t addr_x_, addr_y_, addr_z_;
    uint64_t ukernel_access_size_;
    uint64_t ukernel_count_per_pim_;
    uint32_t *ukernel_;
};


class GemvTransactionGenerator : public TransactionGenerator {
 public:
    GemvTransactionGenerator(uint64_t m,
                             uint64_t n,
                             uint8_t *A,
                             uint8_t *x,
                             uint8_t *y)
        : m_(m), n_(n), A_(A), x_(x), y_(y) {}
    void Initialize() override;
    void SetData() override;
    void Execute() override;
    void GetResult() override;
    void CheckResult() override;

 private:
    void ExecuteBank(int bank);

    uint8_t *A_, *x_, *y_;
    uint8_t *A_T_;
    uint64_t m_, n_;
    uint64_t addr_A_, addr_y_;
    uint64_t ukernel_access_size_;
    uint64_t ukernel_count_per_pim_;
    uint32_t *ukernel_gemv_;
    uint32_t *ukernel_reduce_;
};

class TestTransactionGenerator : public TransactionGenerator {
 public:
    TestTransactionGenerator(uint64_t n,
                             uint8_t *x)
        : n_(n), x_(x) {}
    void Initialize() override;
    void SetData() override;
    void Execute() override;
    void GetResult() override;
    void CheckResult() override;

 private:
    void ExecuteBank(int bank);

    uint8_t *x_;
    uint64_t n_;
    uint64_t addr_x_;
    uint64_t ukernel_access_size_;
    uint64_t ukernel_count_per_pim_;
};


// >> transaction_generator.cc << //
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
    //_mm_mfence();
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

void clflush(uint8_t *mem, size_t size) {
    //for (size_t i = 0; i < size; i += 32)
    //    _mm_clflush(&mem[i]);
    return;
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

            /* >> mmm for clflush optim. */
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
            /* */ // mmm << for clflush optim.

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

            /* >> mmm for clflush optim. */
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
            /* */ // mmm << for clflush optim.

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
	return;
	/*
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
	*/
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

	ukernel_reduce_[0] = 0b10000100100100000000000000000001;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[1]
    ukernel_reduce_[1] = 0b10000100100100000000000000000010;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[2]
    ukernel_reduce_[2] = 0b10000100100100000000000000000011;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[3]
    ukernel_reduce_[3] = 0b10000100100100000000000000000100;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[4]
    ukernel_reduce_[4] = 0b10000100100100000000000000000101;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[5]
    ukernel_reduce_[5] = 0b10000100100100000000000000000110;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[6]
    ukernel_reduce_[6] = 0b10000100100100000000000000000111;   // ADD  GRF_B[0]  GRF_B[0]  GRF_B[7]
    //ukernel_reduce_[7] = 0b00000000000000000000000000000001; // NOP  1
    ukernel_reduce_[7] = 0b01000000100000000000000000000000;   // MOV  BANK      GRF_B[0]
    ukernel_reduce_[8] = 0b00100000000000000000000000000000;   // EXIT
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
	return;
	/*
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
	*/
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

// Initialize variables and ukernel
void TestTransactionGenerator::Initialize() {
    // base address of operands
    addr_x_ = 0;
}

// Write operand data and μkernel to physical memory and PIM registers
void TestTransactionGenerator::SetData() {
    // strided size of one operand with one computation part(minimum)
    uint64_t strided_size = Ceiling(n_ * UNIT_SIZE, SIZE_WORD * NUM_BANK);

    #ifdef debug_mode
    std::cout << "HOST:\tSet input data\n";
    #endif
    // Write input data x to physical memory
    for (int offset = 0; offset < strided_size ; offset += SIZE_WORD)
        TryAddTransaction(addr_x_ + offset, true, x_ + offset);
    return;
}

// Execute PIM computation
void TestTransactionGenerator::Execute() {
    return;
}

// Read PIM computation result from physical memory
void TestTransactionGenerator::GetResult() {
    return;
}

// Calculate error between the result of PIM computation and actual answer
void TestTransactionGenerator::CheckResult() {
    return;
}




//////////////////////////////////////////////
// >> kkm << the real function is here!
// >> kkm << pim_add, mul, linear, bn, lstm!
//////////////////////////////////////////////
// >> shortcut << ssssssssssssssssssssssssssss

torch::Tensor pim_add_forward(
		torch::Tensor input0,
		torch::Tensor input1) {
	std::cout << "PIM ADD !\n";

	int *in0_ = input0.data<int>();
	int *in1_ = input1.data<int>();
	uint8_t *in0 = (uint8_t *)in0_;
	uint8_t *in1 = (uint8_t *)in1_;
	uint8_t *out = (uint8_t *) malloc(sizeof(uint32_t) * 4096);

	TransactionGenerator *tx_generator = new AddTransactionGenerator(4096, in0, in1, out);
	tx_generator->Initialize();
	tx_generator->SetData();
	tx_generator->Execute();
	tx_generator->GetResult();

	return torch::randn({2, 2});
}

torch::Tensor pim_mul_forward(
		torch::Tensor input0,
		torch::Tensor input1) {
	std::cout << "PIM MUL !\n";
	return torch::randn({2, 2});
}

torch::Tensor pim_linear_forward(
		torch::Tensor input,
		torch::Tensor weight) {
	std::cout << "PIM Linear !\n";
	return torch::randn({2, 2});
}

torch::Tensor pim_bn_forward(
		torch::Tensor input,
		torch::Tensor scale,
		torch::Tensor var) {
	std::cout << "PIM BN !\n";
	return torch::randn({2, 2});
}

torch::Tensor pim_lstm_forward(
		torch::Tensor input,
		torch::Tensor hidden_state,
		torch::Tensor weight,
		torch::Tensor bias) {
	std::cout << "PIM LSTM !\n";
	return torch::randn({2, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("add_forward",	&pim_add_forward,	  "PIM_Add forward");
	m.def("mul_forward",	&pim_mul_forward,	  "PIM_Mul forward");
	m.def("linear_forward", &pim_linear_forward,  "PIM_Linear forward");
	m.def("bn_forward",		&pim_bn_forward,	  "PIM_BN forward");
	m.def("lstm_forward",	&pim_lstm_forward,	  "PIM_LSTM forward");
}



