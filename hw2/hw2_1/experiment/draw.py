import matplotlib.pyplot as plt
import os
import numpy as np
'''
with open('./loss_record_basic.txt') as f:
	lines = f.readlines()
values = []
for line in lines:
	index = line.rfind('s')
	values.append(float(line[index+2:len(line)-1]))

values = list(np.mean(np.array(values).reshape(-1, 28), axis=1))

print(len(values))

with open('./loss_record_sche.txt') as f:
	lines = f.readlines()
values1 = []
for line in lines:
	index = line.rfind('s')
	values1.append(float(line[index+2:len(line)-1]))

values1 = list(np.mean(np.array(values1).reshape(-1, 28), axis=1))

print(len(values1))

with open('./loss_record_all.txt') as f:
	lines = f.readlines()
values2 = []
for line in lines:
	index = line.rfind('s')
	values2.append(float(line[index+2:len(line)-1]))

values2 = list(np.mean(np.array(values2).reshape(-1, 28), axis=1))

print(len(values2))

with open('./loss_record_att.txt') as f:
	lines = f.readlines()
values3 = []
for line in lines:
	index = line.rfind('s')
	values3.append(float(line[index+2:len(line)-1]))

values3 = list(np.mean(np.array(values3).reshape(-1, 28), axis=1))

print(len(values3))

#lines = [line.rstrip('\n') for line in open('filename')]
plt.figure()
plt_save_dir = './'
plt_save_img_name = 'loss.png'
plt.plot(range(len(values)), values, color='g',label='Model: basic')
plt.plot(range(len(values1)), values1, color='b',label='Model: basic + scheduling sampling')
plt.plot(range(len(values3)), values3, color='c',label='Model: basic + attention mechanism')
plt.plot(range(len(values2)), values2, color='r',label='Model: basic + scheduling & attention')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.savefig(os.path.join(plt_save_dir,plt_save_img_name))
'''

with open('./bleu_record_basic.txt') as f:
	lines = f.readlines()
values = []
for line in lines:
	index = line.rfind('u')
	values.append(float(line[index+2:len(line)-1]))

with open('./bleu_record_sche.txt') as f:
	lines = f.readlines()
values1 = []
for line in lines:
	index = line.rfind('u')
	values1.append(float(line[index+2:len(line)-1]))

with open('./bleu_record_all.txt') as f:
	lines = f.readlines()
values2 = []
for line in lines:
	index = line.rfind('u')
	values2.append(float(line[index+2:len(line)-1]))

with open('./bleu_record_att.txt') as f:
	lines = f.readlines()
values3 = []
for line in lines:
	index = line.rfind('u')
	values3.append(float(line[index+2:len(line)-1]))

plt.figure()
plt_save_dir = './'
plt_save_img_name = 'bleu.png'
plt.plot(range(len(values)), values, color='g',label='Model: basic')
plt.plot(range(len(values1)), values1, color='b',label='Model: basic + scheduling sampling')
plt.plot(range(len(values3)), values3, color='c',label='Model: basic + attention mechanism')
plt.plot(range(len(values2)), values2, color='r',label='Model: basic + scheduling & attention')
plt.legend(loc='lower right')
plt.ylabel('bleu score')
plt.xlabel('epoch')
plt.grid(True)
plt.savefig(os.path.join(plt_save_dir,plt_save_img_name))
