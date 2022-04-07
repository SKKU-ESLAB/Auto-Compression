#include <iostream>
#include "./args.hxx"
#include "./transaction_generator.h"
#include "./half.hpp"
#include <time.h>

using half_float::half;

// main code to simulate PIM simulator
int main(int argc, const char **argv) {
    srand(time(NULL));
    // parse simulation settings
    args::ArgumentParser parser(
        "PIM-DRAM Simulator.",
        "Examples: \n."
        "./build/pimdramsim3main configs/DDR4_8Gb_x8_3200.ini -c 100 -t "
        "sample_trace.txt\n"
        "./build/pimdramsim3main configs/DDR4_8Gb_x8_3200.ini -s random -c 100");
    args::HelpFlag help(parser, "help", "Display the help menu", {'h', "help"});
    args::ValueFlag<uint64_t> num_cycles_arg(parser, "num_cycles",
                                             "Number of cycles to simulate",
                                             {'c', "cycles"}, 100000);
    args::ValueFlag<std::string> pim_api_arg(
        parser, "pim_api", "PIM API - add, gemv",
        {"pim-api"}, "add");
    args::ValueFlag<uint64_t> add_n_arg(
        parser, "add_n", "[ADD] Number of elements in vector x, y and z",
        {"add-n"}, 4096);
    args::ValueFlag<uint64_t> gemv_m_arg(
        parser, "gemv_m", "[GEMV] Number of rows of the matrix A",
        {"gemv-m"}, 4096);
    args::ValueFlag<uint64_t> gemv_n_arg(
        parser, "gemv_n", "[GEMV] Number of columns of the matrix A",
        {"gemv-n"}, 32);
    args::ValueFlag<uint64_t> test_n_arg(
        parser, "test_n", "[TEST] Number of columns of the matrix A",
        {"test-n"}, 8);


    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    uint64_t cycles = args::get(num_cycles_arg);
    std::string pim_api = args::get(pim_api_arg);
    time_t start, end;
    double result;

    // Initialize modules of PIM-Simulator
    //  Transaction Generator + DRAMsim3 + PIM Functional Simulator
    
    start = clock();
    std::cout << C_GREEN << "Initializing modules..." << C_NORMAL << std::endl;
    TransactionGenerator * tx_generator;

    // Define operands and Transaction generator for simulating computation
    if (pim_api == "add") {
        uint64_t n = args::get(add_n_arg);

        // Define input vector x, y
        uint8_t *x = (uint8_t *) malloc(sizeof(uint16_t) * n);
        uint8_t *y = (uint8_t *) malloc(sizeof(uint16_t) * n);
        // Define output vector z
        uint8_t *z = (uint8_t *) malloc(sizeof(uint16_t) * n);

        // Fill input operands with random value
        for (int i=0; i< n; i++) {
            half h_x = half(rand() / static_cast<float>(RAND_MAX));
            half h_y = half(rand() / static_cast<float>(RAND_MAX));
            ((uint16_t*)x)[i] = *reinterpret_cast<uint16_t*>(&h_x);
            ((uint16_t*)y)[i] = *reinterpret_cast<uint16_t*>(&h_y);
        }

        // Define Transaction generator for ADD computation
        tx_generator = new AddTransactionGenerator(n, x, y, z);
    } else if (pim_api == "gemv") {
        uint64_t m = args::get(gemv_m_arg);
        uint64_t n = args::get(gemv_n_arg);

        // Define input matrix A, vector x
        uint8_t *A = (uint8_t *) malloc(sizeof(uint16_t) * m * n);
        uint8_t *x = (uint8_t *) malloc(sizeof(uint16_t) * n);
        // Define output vector y
        uint8_t *y = (uint8_t *) malloc(sizeof(uint16_t) * m);

        // Fill input operands with random value
        for (int i=0; i< n; i++) {
            half h_x = half(rand() / static_cast<float>(RAND_MAX));
            ((uint16_t*)x)[i] = *reinterpret_cast<uint16_t*>(&h_x);
            for (int j=0; j< m; j++) {
                half h_A = half(rand() / static_cast<float>(RAND_MAX));
                ((uint16_t*)A)[j*n+i] = *reinterpret_cast<uint16_t*>(&h_A);
            }
        }

        // Define Transaction generator for GEMV computation
        tx_generator = new GemvTransactionGenerator(m, n, A, x, y);
    }
    else if (pim_api == "test") {
        uint64_t n = args::get(test_n_arg);

        // Define input vector x
        uint8_t *x = (uint8_t *) malloc(sizeof(uint16_t) * n);

        // Fill input operands with random value
        for (int i=0; i< n; i++) {
            half h_x = half(rand() / static_cast<float>(RAND_MAX));
            ((uint16_t*)x)[i] = *reinterpret_cast<uint16_t*>(&h_x);
        }

        // Define Transaction generator for Test computation
        tx_generator = new TestTransactionGenerator(n, x);
    }
 
    std::cout << C_GREEN << "Success Module Initialize" << C_NORMAL << "\n";
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken: " << (result/CLOCKS_PER_SEC) << "sec \n\n";


    // Initialize variables and ukernel
    start = clock();
    std::cout << C_GREEN << "Initializing severals..." << C_NORMAL << std::endl;
    tx_generator->Initialize();
    std::cout << C_GREEN << "Success Initialize" << C_NORMAL << "\n";
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken: " << (result/CLOCKS_PER_SEC) << "sec \n\n";


    // Write operand data and μkernel to physical memory and PIM registers
    start = clock();
    std::cout << C_GREEN << "Setting Data..." << C_NORMAL << "\n";
    tx_generator->SetData();
    std::cout << C_GREEN << "Success SetData" << C_NORMAL << "\n";
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken: " << (result/CLOCKS_PER_SEC) << "sec \n\n";


    // Execute PIM computation
    start = clock();
    std::cout << C_GREEN << "Executing..." << C_NORMAL << "\n";
    tx_generator->Execute();
    std::cout << C_GREEN << "Success Execute" << C_NORMAL << "\n";
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken: " << (result/CLOCKS_PER_SEC) << "sec \n\n";


    // Read PIM computation result from physical memory
    start = clock();
    std::cout << C_GREEN << "Getting Result..." << C_NORMAL << "\n";
    tx_generator->GetResult();
    std::cout << C_GREEN << "Success GetResult" << C_NORMAL << "\n";
    end = clock();
    result = (double)(end - start);
    std::cout.precision(20);
    std::cout << "time taken: " << (result/CLOCKS_PER_SEC) << "sec \n\n";


    // Calculate error between the result of PIM computation and actual answer
    tx_generator->CheckResult();

    delete tx_generator;

    return 0;
}
