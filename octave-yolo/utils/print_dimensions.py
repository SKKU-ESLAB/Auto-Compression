input_sizes = []
for i in range(320, 640 + 1, 32):
    input_sizes.append(i)

channel = [3, 64, 128, 256, 512, 1024, 512, 256]
layer = 7
pad = 1
stride = 2
kernel = 3

for input_size in input_sizes:
    print('{}x{}x{}'.format(input_size, input_size, channel[0]))

    for i in range(layer):
        output_size = (input_size + 2 * pad - kernel) // stride + 1
        print('{}x{}x{}'.format(output_size, output_size, channel[i + 1]))
        input_size = output_size
    print()
