import matplotlib.pyplot as plt

c1_x = []
c1_y = []

with open('P2_orig.txt', 'r') as f:
    lines = f.readlines()
    for e in lines:
        row = e.split(',')
        c1_x.append(int(row[0]))
        c1_y.append(float(row[1]))
f.close()

c2_x = []
c2_y = []

with open('P2_gamma09.txt', 'r') as f:
    lines = f.readlines()
    for e in lines:
        row = e.split(',')
        c2_x.append(int(row[0]))
        c2_y.append(float(row[1]))
f.close()

c3_x = []
c3_y = []

with open('P2_lr5e2.txt', 'r') as f:
    lines = f.readlines()
    for e in lines:
        row = e.split(',')
        c3_x.append(int(row[0]))
        c3_y.append(float(row[1]))
f.close()

c4_x = []
c4_y = []

with open('P2_tuf200.txt', 'r') as f:
    lines = f.readlines()
    for e in lines:
        row = e.split(',')
        c4_x.append(int(row[0]))
        c4_y.append(float(row[1]))
f.close()


plt.title('Experimenting with DQN hyperparameters\nLearning Curve')
plt.plot(c1_x, c1_y, 'b.-', label='Original')
plt.plot(c2_x, c2_y, 'r.-', label='Gramma = 0.9')
plt.plot(c3_x, c3_y, 'g.-', label='Learning Rate = 5e-2')
plt.plot(c4_x, c4_y, 'c.-', label='Target Update Freq = 200')
plt.legend(loc='best')
plt.show()
