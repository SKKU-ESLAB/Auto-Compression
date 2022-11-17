import os

os.system("g++ -pthread -o mthr_mem_test mthr_mem_trace.cpp")
os.system("cp mthr_mem_test ..")
