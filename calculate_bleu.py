# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 00:11:37 2018

@author: Administrator.Bob
"""
from math import log
from math import exp
from math import e
def calculate_bleu(candidate,reference,weights=(0.25,0.25,0.25,0.25)):
    #candidate=candidate.split(" ")
    #reference=reference.split(" ")
    c=len(candidate)
    r=len(reference)
    bp=exp(1-r/c)
   # print(bp)
    p=[0,0,0,0]
    for gram in list(range(1,min(min(c,r)+1,5))):
        candidate_words=[]
        reference_words=[]
        for i in list(range(0,c-gram+1)):
            candidate_words.append(candidate[i:i+gram])
    #    print(candidate_words)
        for j in list(range(0,r-gram+1)):
            reference_words.append(reference[j:j+gram])
   #     print(reference_words)
        word_list=[x for x in candidate_words if candidate_words.count(x)==1]
        cand_count=0
        ref_count=0
        count=0
        for item in word_list:
            if item in reference_words:
                cand_count=candidate_words.count(item)
                ref_count=reference_words.count(item)
                count+=min(cand_count,ref_count)        
        p[gram-1]=count/(c-gram+1)
       # print(p)
    index=0
    sum_p=0
    for t in p:
        if index==4:
            break
        if t-0>0:
            sum_p+=weights[index]*log(t,e)
            index+=1
        else: 
            index+=1
    bleu=bp*exp(sum_p)
    #print(sum_p)
    return bleu
