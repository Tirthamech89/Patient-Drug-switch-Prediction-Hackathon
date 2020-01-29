#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os


# In[2]:


# load the training set - Do not edit this line
train_transaction_df = pd.read_csv("train_data.csv") #The train dataset name an path should remail unchanged train_data.csv- Do not edit this line
# append the train label.

#load the test set- Do not edit this line
test_transaction_df = pd.read_csv("test_data.csv") #The test dataset name an path should remail unchanged train_data.csv- Do not edit this line


# # Template Function Generation:

# In[3]:


def create_recency_features(data):
    
    patient_v1=pd.DataFrame(data.patient_id.unique())
    patient_v1.columns=['patient_id']
    patient_v1['key']=1

    event_v1=pd.DataFrame(data.event_name.unique())
    event_v1.columns=['event_name']
    event_v1['key']=1
    patient_event_unique=pd.merge(patient_v1,event_v1, on='key')


    specialty_v1=pd.DataFrame(data.specialty.unique())
    specialty_v1.columns=['specialty']
    specialty_v1['key']=1
    patient_specialty_unique=pd.merge(patient_v1,specialty_v1, on='key')


    plan_type_v1=pd.DataFrame(data.plan_type.unique())
    plan_type_v1.columns=['plan_type']
    plan_type_v1['key']=1
    patient_plan_type_unique=pd.merge(patient_v1,plan_type_v1, on='key')
    
    patient=pd.DataFrame(data.patient_id.unique())
    patient.columns=['patient_id']

    pat_event=data.groupby(['patient_id', 'event_name'])['event_time'].min().reset_index()
    patient_event_unique1=pd.merge(patient_event_unique,pat_event, on=['patient_id', 'event_name'],how='left')
    pat_event=None    
    pat_event2=patient_event_unique1.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
    pat_event2= pat_event2.rename_axis(None, axis=1)
    del pat_event

    pat_spcl=data.groupby(['patient_id', 'specialty'])['event_time'].min().reset_index()
    patient_specialty_unique1=pd.merge(patient_specialty_unique,pat_spcl, on=['patient_id', 'specialty'],how='left')
    pat_spcl=None    
    pat_spcl2=patient_specialty_unique1.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
    pat_spcl2= pat_spcl2.rename_axis(None, axis=1)
    del pat_spcl

    pat_pln=data.groupby(['patient_id', 'plan_type'])['event_time'].min().reset_index()
    patient_plan_type_unique1=pd.merge(patient_plan_type_unique,pat_pln, on=['patient_id', 'plan_type'],how='left')
    pat_pln=None    
    pat_pln2=patient_plan_type_unique1.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
    pat_pln2= pat_pln2.rename_axis(None, axis=1)
    del pat_pln

    pat_event2=pat_event2.add_prefix('recency_event_name__')
    pat_event2 = pat_event2.rename(columns = {'recency_event_name__patient_id': "patient_id"})

    pat_spcl2=pat_spcl2.add_prefix('recency_specialty__')
    pat_spcl2 = pat_spcl2.rename(columns = {'recency_specialty__patient_id': "patient_id"})

    pat_pln2=pat_pln2.add_prefix('recency_plan_type__')
    pat_pln2 = pat_pln2.rename(columns = {'recency_plan_type__patient_id': "patient_id"})
    
    final_recency = pd.merge(patient, pat_event2, on='patient_id')
    final_recency = pd.merge(final_recency, pat_spcl2, on='patient_id')
    final_recency = pd.merge(final_recency, pat_pln2, on='patient_id')
    final_recency.fillna(999999999, inplace=True)

    del pat_event2
    del pat_spcl2
    del pat_pln2
    return final_recency
pass


# In[4]:


def create_frequency_features(data):
    
    patient_v1=pd.DataFrame(data.patient_id.unique())
    patient_v1.columns=['patient_id']
    patient_v1['key']=1

    event_v1=pd.DataFrame(data.event_name.unique())
    event_v1.columns=['event_name']
    event_v1['key']=1
    patient_event_unique=pd.merge(patient_v1,event_v1, on='key')


    specialty_v1=pd.DataFrame(data.specialty.unique())
    specialty_v1.columns=['specialty']
    specialty_v1['key']=1
    patient_specialty_unique=pd.merge(patient_v1,specialty_v1, on='key')


    plan_type_v1=pd.DataFrame(data.plan_type.unique())
    plan_type_v1.columns=['plan_type']
    plan_type_v1['key']=1
    patient_plan_type_unique=pd.merge(patient_v1,plan_type_v1, on='key')
    
    
    def frequency(i):
        patient=pd.DataFrame(data.patient_id.unique())
        patient.columns=['patient_id']
        dt=data[data['event_time']<=i].reset_index()
        del dt['index']
    
        dt1=dt.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
        patient_event_unique1=pd.merge(patient_event_unique,dt1, on=['patient_id', 'event_name'],how='left')
        dt1=None
        dt2=patient_event_unique1.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('frequency_'+str(i)+'_event_name__')
        chk2=chk2.rename(columns = {'frequency_'+str(i)+'_event_name__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
        dt1=dt.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
        patient_specialty_unique1=pd.merge(patient_specialty_unique,dt1, on=['patient_id', 'specialty'],how='left')
        dt1=None
        dt2=patient_specialty_unique1.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('frequency_'+str(i)+'_specialty__')
        chk2=chk2.rename(columns = {'frequency_'+str(i)+'_specialty__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
        dt1=dt.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
        patient_plan_type_unique1=pd.merge(patient_plan_type_unique,dt1, on=['patient_id', 'plan_type'],how='left')
        dt1=None
        dt2=patient_plan_type_unique1.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('frequency_'+str(i)+'_plan_type__')
        chk2=chk2.rename(columns = {'frequency_'+str(i)+'_plan_type__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
        return patient

    num_cores = 4     
    results = Parallel(n_jobs=num_cores)(delayed(frequency)(i) for i in range(30,1110,30))
    final_frequency=pd.DataFrame(data.patient_id.unique())
    final_frequency.columns=['patient_id']
    for i in range(0,36):
        final_frequency=pd.merge(final_frequency,results[i], on='patient_id', how='left')
    dt=dt1=dt2=chk2=patient=None
    
    return final_frequency
pass


# In[5]:


def create_normchange_features(data):
    patient_v1=pd.DataFrame(data.patient_id.unique())
    patient_v1.columns=['patient_id']
    patient_v1['key']=1

    event_v1=pd.DataFrame(data.event_name.unique())
    event_v1.columns=['event_name']
    event_v1['key']=1
    patient_event_unique=pd.merge(patient_v1,event_v1, on='key')


    specialty_v1=pd.DataFrame(data.specialty.unique())
    specialty_v1.columns=['specialty']
    specialty_v1['key']=1
    patient_specialty_unique=pd.merge(patient_v1,specialty_v1, on='key')


    plan_type_v1=pd.DataFrame(data.plan_type.unique())
    plan_type_v1.columns=['plan_type']
    plan_type_v1['key']=1
    patient_plan_type_unique=pd.merge(patient_v1,plan_type_v1, on='key')
    
    
    def NormChange(i):
        patient=pd.DataFrame(data.patient_id.unique())
        patient.columns=['patient_id']
    
        data_post = data[data['event_time']<=i].reset_index(drop=True)
        data_pre = data[data['event_time']>i].reset_index(drop=True)
    
        data_post1=data_post.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
        data_post1['feature_value_post'] = data_post1['event_time']/i    
        data_pre1=data_pre.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
        data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)    
        normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'event_name'], how='outer')
        normChange.fillna(0, inplace=True)
        normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
        normChange=normChange[['patient_id','event_name','feature_value']]
        patient_event_unique1=pd.merge(patient_event_unique,normChange, on=['patient_id', 'event_name'],how='left')
        normChange=None        
        dt2=patient_event_unique1.pivot(index='patient_id', columns='event_name', values='feature_value').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('normChange_'+str(i)+'_event_name__')
        chk2=chk2.rename(columns = {'normChange_'+str(i)+'_event_name__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
        data_post1=data_post.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
        data_post1['feature_value_post'] = data_post1['event_time']/i    
        data_pre1=data_pre.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
        data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)    
        normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'specialty'], how='outer')
        normChange.fillna(0, inplace=True)
        normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
        normChange=normChange[['patient_id','specialty','feature_value']]
        patient_specialty_unique1=pd.merge(patient_specialty_unique,normChange, on=['patient_id', 'specialty'],how='left')
        normChange=None
        dt2=patient_specialty_unique1.pivot(index='patient_id', columns='specialty', values='feature_value').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('normChange_'+str(i)+'_specialty__')
        chk2=chk2.rename(columns = {'normChange_'+str(i)+'_specialty__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
        data_post1=data_post.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
        data_post1['feature_value_post'] = data_post1['event_time']/i    
        data_pre1=data_pre.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
        data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)    
        normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'plan_type'], how='outer')
        normChange.fillna(0, inplace=True)
        normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
        normChange=normChange[['patient_id','plan_type','feature_value']]
        patient_plan_type_unique1=pd.merge(patient_plan_type_unique,normChange, on=['patient_id', 'plan_type'],how='left')
        normChange=None
        dt2=patient_plan_type_unique1.pivot(index='patient_id', columns='plan_type', values='feature_value').reset_index()
        dt2=dt2.rename_axis(None, axis=1)
        dt2.fillna(0,inplace=True)    
        chk2=dt2.add_prefix('normChange_'+str(i)+'_plan_type__')
        chk2=chk2.rename(columns = {'normChange_'+str(i)+'_plan_type__patient_id': 'patient_id'})
        patient=pd.merge(patient,chk2, on='patient_id', how='left')
        return patient

    num_cores = 4     
    results = Parallel(n_jobs=num_cores)(delayed(NormChange)(i) for i in range(30,570,30))
    final_normChange=pd.DataFrame(data.patient_id.unique())
    final_normChange.columns=['patient_id']
    for i in range(0,18):
        final_normChange=pd.merge(final_normChange,results[i], on='patient_id', how='outer')
    data_post=data_pre1=data_post1=data_pre=normChange=dt2=chk2=patient=None
    return final_normChange
pass


# In[ ]:


#if __name__ == '__main__':
#    create_recency_features(test_transaction_df)
#    create_frequency_features(test_transaction_df)
#    create_normchange_features(test_transaction_df)


# # Feature Generation:

# In[8]:


recency=create_recency_features(train_transaction_df)
frequency=create_frequency_features(train_transaction_df)
normChange=create_normchange_features(train_transaction_df)


# # Fitness Value Calculation:

# In[22]:


recency_frequency=pd.merge(recency,frequency, on='patient_id', how='left')
recency_frequency_normChange=pd.merge(recency_frequency,normChange, on='patient_id', how='left')
recncy_lst=list(recency_frequency_normChange.columns)
train_labels = pd.read_csv("train_labels.csv")
chk_recency=pd.merge(recency_frequency_normChange,train_labels, on='patient_id', how='left')
fitness_recency=pd.DataFrame()
gbl = globals()
for j in range(1,len(recncy_lst)):
    chk_recency1=chk_recency[['patient_id',recncy_lst[j],'outcome_flag']]
    avg1 = chk_recency1[(chk_recency1['outcome_flag']==1) & (chk_recency1[recncy_lst[j]]!=999999999)][recncy_lst[j]].mean()
    sd1 = chk_recency1[(chk_recency1['outcome_flag']==1) & (chk_recency1[recncy_lst[j]]!=999999999)][recncy_lst[j]].std()
    avg0 = chk_recency1[(chk_recency1['outcome_flag']==0) & (chk_recency1[recncy_lst[j]]!=999999999)][recncy_lst[j]].mean()
    sd0 = chk_recency1[(chk_recency1['outcome_flag']==0) & (chk_recency1[recncy_lst[j]]!=999999999)][recncy_lst[j]].std()
    fitness_recency.at[j-1,'feature_name']=recncy_lst[j]
    fitness_recency.at[j-1,'avg_0']=avg0
    fitness_recency.at[j-1,'avg_1']=avg1
    fitness_recency.at[j-1,'sd_0']=sd0
    fitness_recency.at[j-1,'sd_1']=sd1

def fitness_calculation(data):
    if ((data['sd_0'] == 0 ) and (data['sd_1'] == 0)) and (((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):
        return 9999999999
    elif (((data['sd_0'] == 0 ) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (data['avg_0'] == data['avg_1']):
        return 1
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):
        return ((data['avg_1']/data['sd_1'])/(data['avg_0']/data['sd_0']))
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):
        return 9999999999
    else:
        return 1
    
fitness_recency['coefficient_of_variance'] = fitness_recency.apply(fitness_calculation, axis=1)


# In[23]:


fitness_recency.to_csv('Fitness_Score.csv', index=False)


# In[18]:





# In[19]:





# In[ ]:




