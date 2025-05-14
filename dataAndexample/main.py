import numpy as np
from AdaBoost import AdaBoost
import pandas as pd
def adapt(predictions):
    for i in range(predictions.shape[0]):
        if predictions[i]==-1:
            predictions[i]=0
def main():
    fold_num = 10
    base_list = [1, 5, 10, 100]
    target_dir = 'my_experiments/'
    data = np.genfromtxt('data.csv',dtype=float,delimiter=',')
    fold_length = data.shape[0]/fold_num
    label = np.genfromtxt('targets.csv',dtype=int,delimiter=',')
    for base_num in base_list:
        classifier = AdaBoost(n_estimators=base_num)
        for fold in range(1,fold_num+1):
            x_train = np.concatenate(data[(fold-2)*fold_length:(fold-1)*fold_length] ,
                                     data[(fold)*fold_length:])if not fold==1 else data[fold*fold_length:]
            y_train = np.concatenate(label[(fold-2)*fold_length:(fold-1)*fold_length] ,
                                     label[(fold)*fold_length:])if not fold==1 else label[(fold)*fold_length]
            x_test = data[(fold-1)*fold_length:fold*fold_num]
            classifier.fit(x_train,y_train)
            predictions = adapt(classifier.predict(x_test))
            start_index = (fold-1)*fold_length+1
            df= pd.DataFrame(predictions,index=range(start_index,start_index+predictions.shape[0]))
            df.to_csv(target_dir+'base%d_fold%d.csv' % (base_num, fold))


if __name__ == '__main__':
    main()