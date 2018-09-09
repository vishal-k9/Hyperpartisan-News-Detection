
cnt1=0
cnt2=0
with open("trial_train.txt","rb") as f1:
	for row in f1:
		col= row.strip().split()
		if(col[2].strip()=="true"):
			cnt1+=1
		else:
			cnt2+=1


print cnt1,cnt2
