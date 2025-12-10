import numpy as np
data = np.loadtxt('./output_test_ode.txt', usecols=(0, 1, 2))
# print (data)

import matplotlib.pyplot as plt

plt.plot(data[:,0], data[:,1], label='position')
plt.plot(data[:,0], data[:,2], label='velocity')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Mass-Spring System Time Evolution')
plt.legend()
plt.grid()
plt.show()


plt.plot(data[:,1], data[:,2], label='phase plot')
plt.xlabel('position')
plt.ylabel('velocity')
plt.title('Mass-Spring System Phase Plot')
plt.legend()
plt.grid()
plt.show()

