import matplotlib.pyplot as plt
import numpy as np 

alpha_records = list(np.arange(-1.0, 2.5, 0.5))

train_loss_record = [1.150884738485019, 2.3540019112229347, 0.02004321910295937, 0.011061786843686017, 0.017476922628642106, 14.449872374216715, 14.281303994496664]
test_loss_record = [0.7293439184824638, 1.30187077554067, 0.08377377572709793, 0.06936186309908889, 0.05840066910071448, 14.375997774759929, 14.40957709757487]
train_acc_record = [0.9285625, 0.8539375, 0.9961458333333333, 0.9975208333333333, 0.9960416666666667, 0.1035, 0.11395833333333333]
test_acc_record = [0.95475, 0.9191666666666667, 0.9898333333333333, 0.9898333333333333, 0.9895, 0.10808333333333334, 0.106]
    
ax1 = plt.plot()
plt.plot(alpha_records, train_loss_record, linestyle= '-', color= 'blue', label= 'train_loss')
plt.plot(alpha_records, test_loss_record, linestyle= '--', color= 'blue', label= 'test_loss')
plt.xlabel('alpha')
plt.ylabel('cross entropy',color='blue')
legend = plt.legend(loc= 'upper left')
legend.get_frame().set_alpha(0.5)

ax2 = plt.gca().twinx()
plt.plot(alpha_records, train_acc_record, linestyle= '-', color= 'red', label= 'train_acc')
plt.plot(alpha_records, test_acc_record, linestyle= '--', color= 'red', label= 'test_acc')
plt.ylabel('accuracy', color='red')
legend = plt.legend(loc= 'upper right')
legend.get_frame().set_alpha(0.5)

plt.savefig('fig1_3_3_1.png')