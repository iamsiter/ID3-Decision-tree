#Decision Tree implementation using ID3
#sahils (Sahil Sachdeva)
#vmanocha (Varun Manocha)

import yaml
import csv
import sys
import math

#Splits the table for given attribute and its domain values
def getSplitTable(datacopy,index,domval):
    returnData=[]
    for d in datacopy:
        if d[index]==domval:
            dCopy=d[:]
            dCopy.remove(domval)
            returnData.append(dCopy)
    return returnData



#Calculates the entropy
def entropy(dictEntropy,totalRow):
    entropy=0
    for key,val in dictEntropy.items():
        sum=0
        temp=0
        for keyInnerDict,valInnerDict in val.items():
            sum+=valInnerDict
        for keyInnerDict,valInnerDict in val.items():
            if(valInnerDict==0):
			    continue #check afterwards to better
            temp+=-(float(valInnerDict)/sum)*math.log(float(valInnerDict)/sum,2)
        entropy+=temp*(float(sum)/totalRow)
    return entropy


#Returns the domain values 
def getDomainValues(data,index):
    mylist=[]
    for row in data:
        mylist.append(row[index])
    return list(set(mylist))   


#Selects the next best attribute based on information gain
def selectBestAttr(data,attr,prediction):
    attr=attr[:-1]
    target=attr[-1]
    index=0
    totalRow=0
    ans=""
    domainTarget=getDomainValues(data,-1)
    dictPredictClass={}
    for entry in data:
        if (entry[-1] in dictPredictClass):
            dictPredictClass[entry[-1]] += 1.0
        else:
            dictPredictClass[entry[-1]]  = 1.0
    entropyTotal=0
    for key,val in dictPredictClass.items():
        entropyTotal+= -1*(float(val)/sum(dictPredictClass.values()))*math.log(float(val)/sum(dictPredictClass.values()),2)
    max=-sys.maxsize - 1
    for a in attr:
        domainVals=getDomainValues(data,index)
        tempDict={}
        for dval in domainVals:
            domainDict= dict.fromkeys(domainTarget, 0)
            tempDict[dval]=domainDict
        for d in data:
            tar=d[-1];
            tempDict[d[index]][tar]+=1;
        tempMax=entropyTotal-entropy(tempDict,len(data))
        if(tempMax>max):
            max=tempMax
            ans=a
        index+=1;
    return ans
           
    
    
#Main function to build tree
def buildDecisionTree(attributes,data):
    prediction=attributes[-1]
    dataCopy=data[:]
    if(len(attributes)-1<=0):
        from collections import Counter
        c=[]
        for row in dataCopy:
            c.append(row[-1])
        maxCounter=Counter(c)
        k,v=maxCounter.most_common()[0]
        return k
   
    if len({c[-1] for c in dataCopy}) == 1:
        return dataCopy[0][-1]
   
    
    nextClass=selectBestAttr(dataCopy,attributes,prediction)
    domainVals=getDomainValues(dataCopy,attributes.index(nextClass))    
    index=attributes.index(nextClass)
 
    tree={nextClass:{}}
    for domVal in domainVals:
        splitData=getSplitTable(dataCopy,index,domVal)
        attrCopy = attributes[:]
        attrCopy.remove(nextClass)
        tree[nextClass][domVal]=buildDecisionTree(attrCopy,splitData)
    return tree;

#Function to predict class for the query
def classify(tree,query):
    if type(tree) is dict:
        root=list(tree.keys())[0]
        if root in query:
            return classify(tree[root][query[root]],query)
        else:
            return root
    else:
        return tree
		
#Function to print the tree for input dictionary      
def treePrint(t,s):
    if not isinstance(t,dict) and not isinstance(t,list):
        print ("  "*s+"-->"+str(t))
    else:
        for key in t:
            print ("  "*s+str(key)+":")
            if not isinstance(t,list):
                treePrint(t[key],s+1)    
				
#Read the training data    
dataFile = open("training_data","r")
data = dataFile.read()
tData = [[ele.strip(" ") for ele in row.split(", ")] for row in data.split("\n")]
del(tData[-1])

attributes=["Occupied","Price","Music","Location","VIP","Favorite Beer","Enjoy"]
tree = buildDecisionTree(attributes,tData)
treePrint(tree,0)

query={'Occupied':'Moderate','Price':'Cheap','Music':'Loud','Location':'City-Center','VIP':'No','Favorite Beer':'No'}
print("\nPredicted Output For Test case specified is")
print(classify(tree,query))