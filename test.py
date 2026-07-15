import numpy as np
indexes=[]
cpg_sorted=[0,1,2,3,4]
non_cpg_sorted=[4,0,1,3,2]
i=0
j=0
cpg_non_cpg_choices=[]
random=[]
for x in range(len(cpg_sorted)):
    choice = np.random.choice([0, 1])
    random.append(choice)
    if choice==0:
        if i>=len(cpg_sorted): #only intems in j list left, so have to choose from there
            choosen_index= non_cpg_sorted[j]
            j+=1
            cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
            cpg_non_cpg_choices.append(1)   #1 is added non cpg, 0 is added cpg
        else:
            choosen_index=cpg_sorted[i] #chosen index is the next smallest cpg
            i+=1   
            non_cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
            cpg_non_cpg_choices.append(0)

    else:
        if j>=len(non_cpg_sorted): #only intems in i list left, so have to choose from there
            choosen_index= cpg_sorted[i]
            i+=1 
            non_cpg_sorted.remove(choosen_index)
            cpg_non_cpg_choices.append(0)
        else:
            choosen_index=non_cpg_sorted[j]
            j+=1
            cpg_sorted.remove(choosen_index)
            cpg_non_cpg_choices.append(1)

    indexes.append(choosen_index)
    
print(i,j, indexes, cpg_non_cpg_choices,random)
