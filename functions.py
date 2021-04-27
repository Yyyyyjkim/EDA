#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import re
import warnings
from scipy.stats import mode
import matplotlib.font_manager as fm
font_path = 'NanumGothicBold.ttf'
fontprop = fm.FontProperties(fname=font_path)
warnings.filterwarnings(action='ignore')


# In[ ]:


# 수치형 단일변수
def plot_num(x_list, data):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.plot(data[x_list[i]],'.')
        plt.title(x_list[i]+' plot', fontproperties=fontprop, size=15)
        plt.ylabel(x_list[i])
        plt.subplot(132)
        sns.histplot(data=data, x=x_list[i])
        plt.title(x_list[i]+' histogram', fontproperties=fontprop, size=15)
        plt.subplot(133)
        sns.kdeplot(data=data, x=x_list[i])
        plt.title(x_list[i]+' density plot', fontproperties=fontprop, size=15)
        plt.show()


# In[ ]:


# 범주형 단일변수
def plot_cat(x_list, data):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))
        plt.subplot(121)
        sns.countplot(data=data, x=x_list[i], color='#5586EB', order=sorted(list(set(data[x_list[i]]))))
        plt.title(x_list[i]+' count plot', fontproperties=fontprop, size=15)
        plt.ylabel(x_list[i])
        plt.subplot(122)
        sns.countplot(data=data, x=x_list[i], color='#5586EB', order=data[x_list[i]].value_counts().index)
        plt.title(x_list[i]+' count plot (빈도 높은 순)', fontproperties=fontprop, size=15)
        plt.show()


# In[ ]:


# 시계열 단일변수
def plot_date(x_list, data):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))
        plt.subplot(121)
        count = pd.DataFrame(np.ones(data.shape[0]))
        time_data = count.groupby(data[x_list[i]]).sum().rename(columns={0:'count'}).reset_index()
        sns.lineplot(data=time_data, x=x_list[i], y='count')
        plt.xticks(rotation=90)
        plt.title(x_list[i]+'별 count plot', fontproperties=fontprop, size=15)
        
        plt.subplot(122)
        month = [str(i)[:7].replace('-','') for i in data[x_list[i]]]
        sns.countplot(x=month, color='#5586EB')
        plt.xticks(rotation=90)
        plt.title(x_list[i]+' month별 count plot', fontproperties=fontprop, size=15)
        
        plt.show()


# In[ ]:


# num2num
def plot_num2num(x_list, target, data, date=None):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,10))
        plt.subplots_adjust(hspace=0.3)
        
        plt.subplot(2,2,1)
        sns.scatterplot(data=data, x=x_list[i], y=target, alpha=0.3)
        plt.title(x_list[i]+' vs '+target+' scatter plot', fontproperties=fontprop, size=15)
        
        plt.subplot(2,2,2)
        sns.kdeplot(data=data, x=x_list[i], y=target, shade=True)
        plt.title(x_list[i]+' vs '+target+' density plot', fontproperties=fontprop, size=15)
        
        plt.subplot(2,2,3)
        sns.lineplot(data=data, x=x_list[i], y=target)
        plt.title(x_list[i]+' vs '+target+' line plot', fontproperties=fontprop, size=15)
        
        if date!=None:
            ax1 = plt.subplot(2,2,4)
            ax2 = ax1.twinx()
            l1 = sns.lineplot(data=data, x=date, y=x_list[i], color='blue', ax=ax1)
            l2 = sns.lineplot(data=data, x=date, y=target, color='red', ax=ax2)
            ax1.legend(loc='upper left', labels=[x_list[i]])
            ax2.legend(loc='upper right', labels=[target])
            plt.title(date+' 별 '+x_list[i]+' vs '+target+' line plot', fontproperties=fontprop, size=15)

        plt.show()


# In[1]:


# cat2num

def plot_cat2num(x_list, target, data, round_num=2):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,10))
        plt.subplots_adjust(hspace=0.5)
        
        plt.subplot(2,3,1)
        sns.countplot(data=data, x=x_list[i], color='#5586EB', order=sorted(list(set(data[x_list[i]]))))
        plt.xticks(rotation=90)
        plt.title(x_list[i]+' 분포 (count plot)', fontproperties=fontprop, size=15)
        
        plt.subplot(2,3,2)
        sns.barplot(data=data, x=x_list[i], y=target, color='#5586EB', order=sorted(list(set(data[x_list[i]]))))
        plt.xticks(rotation=90)
        plt.title(x_list[i]+'별 '+target+' 분포 (bar plot)', fontproperties=fontprop, size=15)
        
        plt.subplot(2,3,3)
        sns.heatmap(pd.crosstab(round(data[target],round_num), data[x_list[i]]),cmap='Blues')
        plt.gca().invert_yaxis()
        plt.title(x_list[i]+'별 '+target+' 분포 (heatmap)', fontproperties=fontprop, size=15)
        
        plt.subplot(2,3,4)
        sns.boxplot(data=data, x=x_list[i], y=target, color='#5586EB', order=sorted(list(set(data[x_list[i]]))))
        plt.xticks(rotation=90)
        plt.title(x_list[i]+'별 '+target+' 분포 (box plot)', fontproperties=fontprop, size=15)
        
        plt.subplot(2,3,5)
        sns.boxplot(data=data, x=x_list[i], y=target, color='#5586EB', order=sorted(list(set(data[x_list[i]]))),
                   showfliers=False)
        plt.xticks(rotation=90)
        plt.title(x_list[i]+'별 '+target+' 분포 (box plot, without outliers)', fontproperties=fontprop, size=15)
        
        plt.show()


# In[2]:


# date2num

def plot_date2num(x_list, target, data):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))

        plt.subplot(1,2,1)
        sns.lineplot(data=data, x=x_list[i], y=target)
        plt.xticks(rotation=90)
        plt.title(x_list[i]+' 별 '+target+' 분포 (line plot)', fontproperties=fontprop, size=15)

        plt.subplot(1,2,2)
        month = [str(i)[:7].replace('-','') for i in data[x_list[i]]]
        sns.barplot(x=month, y=data[target], color='#5586EB', 
                        order=sorted(list(set(month))))
        plt.xticks(rotation=90)
        plt.title(x_list[i]+' 월별 '+target+' 분포 (bar plot)', fontproperties=fontprop, size=15)

        plt.show()


# In[4]:


# num2group

def plot_num2group(x_list, target, data, bw=0.5, binlist=None, shade=False, hue_order=None):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))
        
        if binlist==None:
            
            plt.subplot(121)
            sns.kdeplot(data=data, x=x_list[i], hue=target, bw=bw, shade=shade, hue_order=hue_order)
            plt.title(target+' 별 '+x_list[i]+' density plot', fontproperties=fontprop, size=15)

            plt.subplot(122)
            sns.histplot(data=data, x=x_list[i], hue=target, hue_order=hue_order)
            plt.title(target+' 별 '+x_list[i]+' histogram', fontproperties=fontprop, size=15)
        
        else:
            
            plt.subplot(121)
            sns.kdeplot(data=data, x=x_list[i], hue=target, bw=bw, shade=shade, hue_order=hue_order)
            plt.title(target+' 별 '+x_list[i]+' density plot', fontproperties=fontprop, size=15)

            plt.subplot(122)
            sns.histplot(data=data, x=x_list[i], hue=target, binwidth=binlist[i], hue_order=hue_order)
            plt.title(target+' 별 '+x_list[i]+' histogram', fontproperties=fontprop, size=15)

        plt.show()


# In[ ]:


# cat2group

def plot_cat2group(x_list, target, data, annot=False, order=None):
    for i in range(len(x_list)):
        
        cross_data = pd.crosstab(data[target], data[x_list[i]])
        x_sum = pd.crosstab(data[target], data[x_list[i]]).sum(axis=1)
        target_sum = pd.crosstab(data[target], data[x_list[i]]).sum(axis=0)
        cross_x = pd.crosstab(data[target], data[x_list[i]]).apply(lambda x: x/x_sum, axis=0)
        cross_target = pd.crosstab(data[target], data[x_list[i]]).apply(lambda x: x/target_sum, axis=1)
        
        plt.figure(figsize=(20,5))
        if order == None:
            plt.subplot(131)
            sns.heatmap(cross_data, annot=annot, fmt='d')
            plt.title(x_list[i]+' 별 '+target+' 분포', fontproperties=fontprop, size=15)
            
            plt.subplot(132)
            sns.heatmap(cross_x, annot=annot, fmt='.2f')
            plt.title(x_list[i]+' 별 '+target+' 분포 (각 '+target+' level 합 = 1)', fontproperties=fontprop, size=15)

            plt.subplot(133)
            sns.heatmap(cross_target, annot=annot, fmt='.2f')
            plt.title(x_list[i]+' 별 '+target+' 분포 (각 '+x_list[i]+' level 합 = 1)', fontproperties=fontprop, size=15)

        else :
            plt.subplot(131)
            sns.heatmap(cross_data.loc[order], annot=annot, fmt='d')
            plt.title(x_list[i]+' 별 '+target+' 분포', fontproperties=fontprop, size=15)
            
            plt.subplot(132)
            sns.heatmap(cross_x.loc[order], annot=annot, fmt='.2f')
            plt.title(x_list[i]+' 별 '+target+' 분포 (각 '+target+' level 합 = 1)', fontproperties=fontprop, size=15)

            plt.subplot(133)
            sns.heatmap(cross_target.loc[order], annot=annot, fmt='.2f')
            plt.title(x_list[i]+' 별 '+target+' 분포 (각 '+x_list[i]+' level 합 = 1)', fontproperties=fontprop, size=15)


# In[ ]:


# date2group

def plot_date2group(x_list, target, data):
    for i in range(len(x_list)):
        plt.figure(figsize=(20,5))
        colname = data.columns.drop([x_list[i],target])[0]
        count_data = data[[x_list[i], target, colname]].groupby([x_list[i], target]).count()            .rename(columns={colname: 'count'}).reset_index()
        sns.lineplot(data=count_data, x=x_list[i], y='count', hue=target)
        plt.title(target+' 별 '+x_list[i]+' 별 count 분포', fontproperties=fontprop, size=15)
        
        plt.show()

