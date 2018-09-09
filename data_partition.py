import numpy as np
from numpy.random import multinomial

n_total_samples =199981  # or whatever it is

indices = np.arange(n_total_samples)
inds_split = multinomial(n=1,
                         pvals=[0.8, 0.1, 0.1],
                         size=n_total_samples).argmax(axis=1)

train_inds = indices[inds_split==0]
test_inds  = indices[inds_split==1]
dev_inds   = indices[inds_split==2]

print len(train_inds) 
print len(test_inds) 
print len(dev_inds) 

print (train_inds) 
print (test_inds) 
print (dev_inds) 


# a=[idx for idx in xrange(10)]
# # print a
# for idx in xrange(10):
# 	if idx in train_inds:
# 		print "train"
# 	# print a[idx]
# 	# print idx

with open("trial_data.txt", "rb") as f, open("trial_train.txt","wb") as f1, open("trial_dev.txt","wb") as f2,  open("trial_test.txt","wb") as f3: 
	i=0
	for row in f:
		if i in train_inds:
			f1.write(row)
		elif i in test_inds:
			f3.write(row)
		else:
			f2.write(row)	
		i+=1		

