def preprocess():
feature_count = X_train.shape[1]
W = np.zeros((101,2,feature_count))
for i in range(feature_count):
    feature = X_train_magnified[:,i]  
    for num in range(101):
      index = np.argwhere(feature == num)
      target_val,counts=np.unique(y_train.to_numpy()[index],return_counts=True)
      if len(counts) == 0:
        counts = np.zeros(2)
      if len(counts) < 2 and len(counts)!=0:
        tmp = np.zeros(2)
        for x,y in zip(target_val,counts):
            tmp[x] = y
        counts = tmp 
      W[num,:,i] = counts

P = np.zeros((101,feature_count))
for i in range(feature_count):
    for j in range(101):
        if W[j,:,i].sum() == 0:
            P[j,i] = 0
        else:
            P[j,i] = W[j,:,i].max()/W[j,:,i].sum()

P_scaled = scaler.fit_transform(P)

S = np.zeros(feature_count)
for i in range(feature_count):
    S[i] = np.sum(W[:,:,i].sum(axis=1) * P_scaled[:,i])/np.sum(W[:,:,i].sum(axis=1))

S_prime = (t*(k/(t-np.power(S,2))))-k

for i in range(feature_count):
    X_train.iloc[:,i]= X_train.iloc[:,i] * S_prime[i]
    X_test.iloc[:,i]= X_test.iloc[:,i] * S_prime[i]
