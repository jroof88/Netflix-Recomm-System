import sys
import numpy as np
import math

#Load training data into matrix
trainData = np.loadtxt("train.txt")


def cosine_similarity(train, test): #Cosine sim function
    assert(len(train)==len(test))
    numerator = 0
    trainSquared = 0
    testSquared = 0
    for i in range(0, len(train)):
        if(train[i] != 0 and test[i] != 0):
            numerator += train[i]*test[i]
            trainSquared += pow(train[i],2)
            testSquared += pow(test[i],2)
    
    denominator = float(math.sqrt(trainSquared))*float(math.sqrt(testSquared))

    if(denominator == 0):
        return 0
    else:
        return numerator/denominator
    
def runCosineSim(num): #Main driver for cosine sim
    testData = np.loadtxt("test" + str(num) + ".txt")
    f = open("results" + str(num) + ".txt", "w")
    currUserData = np.zeros(1000, dtype=np.int)
    i = 0
    while(i < len(testData)):
        if(testData[i][2] == 0):
            similarities = {}
            tmpCosine = 0
            for j in range(0, len(trainData)):
                tmpCosine = cosine_similarity(trainData[j], currUserData)
                if(tmpCosine != 0):
                    similarities[j] = tmpCosine
            currUser = testData[i][0]
            for k in range(i, len(testData)):
                if(testData[k][0] != currUser):
                    break
                numerator = 0
                denominator = 0
                idx = int(testData[k][1])-1
                rating = 0
                for key, value in similarities.items():
                    if(trainData[key][idx] != 0):
                        print(trainData[key][idx])
                        numerator += value*trainData[key][idx]
                        denominator += value
                if(denominator != 0):
                    rating = round(numerator/denominator)
                if(rating == 0):
                    t = str(int(testData[k][0])) + " " + str(int(testData[k][1])) + " 3\n"
                    f.write(t)
                else:
                    t = str(int(testData[k][0])) + " " + str(int(testData[k][1])) + " " + str(int(rating)) + "\n"
                    f.write(t)
                #print(rating)
                i += 1
            currUserData = np.zeros(1000, dtype=np.int)
        else:    
            currUserData[int(testData[i][1]-1)] = int(testData[i][2]) 
            i += 1
                    
if __name__ == '__main__':
    if(sys.argv[1] == "5"):
        runCosineSim(5)
    elif(sys.argv[1] == "10"):
        runCosineSim(10)
    elif(sys.argv[1] == "20"):
        runCosineSim(20)
    else:
        print("Invalid System Arguement. Please run script again")
        exit()
    
        
