from skmultilearn.model_selection import IterativeStratification
import sys 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 
#import matplotlib.pyplot as plt 
#from sklearn.model_selection import StratifiedKFold, KFold
#import pandas as pd 


#get the boundaries to use to split into chunks with increasing value 
def slice_index(array, n_chunks ):
    partitioned_list = np.array_split(np.sort(array), n_chunks)
    return [i[-1] for i in partitioned_list]
    

def multilabel_matrix_maker(df, binary_cols = None, multiclass_cols = None , continuous_cols = None , n_chunks = None) :
    """
    returns matrix that will be used for multilabel, taking into account columns that are either multiclass or continuous
    * df : the dataframe to be split
    * binary_cols : LIST of cols (str)just cols that will be used (binarized)
    * multiclass_cols : LIST of the cols (str) that are multi class
    * continuous_cols : LIST of the cols (str) that will be split (continouous)
    * n_chunks : if using continouous cols are used, how many split?
    
    outputs matrix that has binarized binarized for all columns (only needs to be used during iskf to get the indices)
    """
    if binary_cols == multiclass_cols == continuous_cols == None : #i.e. if all are None
        raise ValueError("at least one of the cols have to be put.. currently all cols are selected as None")
    
    
    #checking if NaN exist => raise error (sanity check)
    for col_list in [binary_cols, multiclass_cols, continuous_cols]:
        if col_list : 
            if df[col_list].isnull().values.any():
                raise ValueError(f"column {col_list}'s element in df NaN. most likely your provided df had some NaN in columns that you are wanting to do iskf on")    
    
    #now adding binarized columns for each column types and aggregating them into total_cols
    total_cols = []
    if binary_cols : 
        for col in binary_cols :
            print(col)
            total_cols.append(df[col].values) #or single []?  ([[]] : df 로 만드는 것, [] : series로 만듬) 
            
    if multiclass_cols :
        for col in multiclass_cols : 
            df_col = df[[col]] #[[]] not [] because of dims 
            ohe = OneHotEncoder()
            ohe.fit(df_col)
            binarized_col = ohe.transform(df_col).todense() 
            total_cols.append(binarized_col)

    if continuous_cols : 
        if not n_chunks : 
            raise ValueError("n_chunks must be provided when runing continouous cols")
        else : 
            for col in continuous_cols:
                array = df[col].values
                boundaries = slice_index(array, n_chunks)  
                i_below = -np.infty
                for i in boundaries:
                    extracted_df = (df[col]>i_below) & (df[col]<=i) 
                    i_below = i #update i_below
                    total_cols.append(extracted_df.values.astype(float))     
    
    #adding all together,
    final_arr = np.column_stack(total_cols)
    
    return final_arr


"""
#example of usage : 
DEBUG = False
from skmultilearn.model_selection import IterativeStratification
kf = StratifiedKFold(n_splits=5)
ikf = IterativeStratification(n_splits=5, order=50, random_state=np.random.seed(seed = 0)) #increasing order makes it similar

binary_col2view = ['sex', "Unspecified.Bipolar.and.Related.Disorder.x"]
multiclass_col2view = [] #['race.ethnicity']
continuous_col2view = ['age'] #['age'] #['age'] #없애고 싶으면 [] 로 하기 
###===setting complete====###



##config에서 넣어줄때, 종류를 미리 나눠서 ㄴ허어줘야할듯 

#col2view = binary_col2view + multiclass_col2view + continuous_col2view
col2view = binary_col2view + multiclass_col2view
#print(col2view)
print(calc_prop(label_tv[col2view].values)) if DEBUG else None




from iter_strat import multilabel_matrix_maker as maker
#floatized_arr = np.array(label_tv[i2view].values, dtype = float)
floatized_arr = maker(label_tv,binary_cols= binary_col2view, 
                                        multiclass_cols= multiclass_col2view,
                                       continuous_cols=continuous_col2view, n_chunks=3)#, contiuous_cols=['age', 'BMI'], n_chunks=)


set_thing = set(label_tv[col2view].values.flatten()) #un split된 상태에서의 set을 써야함
thing = []
haha = []
#single label
#label_name = ['Unspecified.Bipolar.and.Related.Disorder.x'] #sex로 고정
#for FOLD, (train_idx, valid_idx) in enumerate(kf.split(label_tv, label_tv[label_name])): 
#multilabel
for FOLD, (train_idx, valid_idx) in enumerate(ikf.split(floatized_arr, floatized_arr)): 
    print(f"===FOLD : {FOLD}===")
    train = label_tv.iloc[train_idx]
    valid = label_tv.iloc[valid_idx]

    if DEBUG : 
        print("with training")
        print(calc_prop_change(train[col2view].values, label_tv[col2view].values, set_thing = set_thing))
        
        print("with validation")
        print(calc_prop_change(valid[col2view].values, label_tv[col2view].values, set_thing = set_thing))
    #thing.append(valid[col2view[-2]])
    
    #thing.append(valid[binary_col2view[1]])
    #thing.append(valid[multiclass_col2view[0]])
    thing.append(valid[continuous_col2view[0]])
    haha.append(calc_prop_change(valid[col2view].values, label_tv[col2view].values, set_thing = set_thing))
    
plt.hist(thing,bins = 3, label = [i for i in range(5)])
plt.legend()
plt.show()


##continuous variables checking

##sometimes, the validation set might not have the same size, even if it can become like that  (so, for loop으로 통계계산)
print([i.mean() for i in thing])
print(np.array([i.mean() for i in thing]).std())
print("haha")

##discrete stuff checking
haha_arr = np.array(haha, dtype = float)
print(np.sqrt(np.square(haha_arr).sum(axis=0)/5))

"""