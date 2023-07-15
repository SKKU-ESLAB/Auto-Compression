import math 


alpha_l_in = 4
alpha_m_in = 2
alpha_h_in = 1

alpha_l_out = 4
alpha_m_out = 2
alpha_h_out = 1

in_channels=256
out_channels=256

l2l = [(alpha_l_in * in_channels), (alpha_l_out * out_channels)]

m2m =[(alpha_m_in * in_channels), (alpha_m_out * out_channels)]

h2h =[(alpha_h_in * in_channels), (alpha_h_out * out_channels)]

print (h2h)
print (m2m)
print (l2l)


