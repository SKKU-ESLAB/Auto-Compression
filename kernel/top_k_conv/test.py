import tvm
from tvm import te
import numpy as np

target = "llvm"

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
s = te.create_schedule(C.op)
f = tvm.build(s, [A, B, C], target=target, target_host=target, name="simple_add")

print(tvm.lower(s, [A, B, C], simple_mode=True))

a = np.random.randn(3).astype("float32")
b = np.random.randn(3).astype("float32")
c = np.random.randn(3).astype("float32")
tvm_a = tvm.nd.array(a)
tvm_b = tvm.nd.array(b)
tvm_c = tvm.nd.array(c)
print(a)
print(b)
print(c)

print()
print(a+b)
f(tvm_a, tvm_b, tvm_c)
print(tvm_c)
