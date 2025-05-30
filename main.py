import numpy as np
import AdaBoost7
import AdaBoost9
import pandas as pd
import argparse
import os

def adapt(predictions):
    for i in range(predictions.shape[0]):
        if predictions[i]==-1:
            predictions[i]=0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t',type=str,required=False,default='experiments/')
    parser.add_argument('--m',type=int,required=False,default=0)
    args = parser.parse_args()
    fold_num = 10
    base_list = [1, 5, 10, 100]
    target_dir = args.t
    m = args.m
    if not target_dir[-1]=='/':
        target_dir+='/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    data = np.genfromtxt('data/data.csv',dtype=float,delimiter=',')
    fold_length = int(data.shape[0]/fold_num)
    label = np.genfromtxt('data/targets.csv',dtype=int,delimiter=',')
    for base_num in base_list:
        print(f'base num = {base_num}.\nNow begin to train and test')
        for fold in range(1,fold_num+1):
            if m==0:
                classifier =AdaBoost7.AdaBoost(n_estimators=base_num)
            elif m==1:
                classifier = AdaBoost9.AdaBoost(base_num)
            else:
                print('please input correct command line argument,--m 0 or --m 1')
                return

            print(f'base{base_num}fold{fold},please waiting...')
            x_train = np.concatenate((data[:(fold-1)*fold_length] ,
                                     data[(fold)*fold_length:]),axis=0)#if not fold==1 else data[fold*fold_length:]
            y_train = np.concatenate((label[:(fold-1)*fold_length] ,
                                     label[(fold)*fold_length:]),axis=0)#if not fold==1 else label[(fold)*fold_length:]
            x_test = data[(fold-1)*fold_length:fold*fold_length]
            classifier.fit(x_train,y_train)
            predictions = classifier.predict(x_test)
            adapt(predictions)
            start_index = (fold-1)*fold_length+1
            df= pd.DataFrame(predictions,index=range(start_index,start_index+predictions.shape[0]))
            df.to_csv(target_dir+'base%d_fold%d.csv' % (base_num, fold))


if __name__ == '__main__':
    main()
