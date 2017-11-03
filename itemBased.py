import sys
import numpy as np
import math

#Load training data into matrix
trainDataImport = np.loadtxt("train.txt")
print(trainDataImport.shape) #Check shape
trainData = np.transpose(trainDataImport)
print(trainData.shape) #Check shape to verify transpose


def cosine_similarity(train, test, userAverage): #Adjusted cosine sim function
    assert(len(train)==len(test))
    numerator = 0
    trainSquared = 0
    testSquared = 0
    for i in range(0, len(train)):
        if(train[i] != 0 and test[i] != 0):
            numerator += (train[i]-userAverage)*(test[i]-userAverage)
            trainSquared += pow(train[i]-userAverage,2)
            testSquared += pow(test[i]-userAverage,2)
    
    denominator = float(math.sqrt(trainSquared))*float(math.sqrt(testSquared))

    if(denominator == 0):
        return 0
    else:
        return numerator/denominator
        
def runItemBased(num): #Main driver for item based
    testData = np.loadtxt("test" + str(num) + ".txt")
    f = open("results" + str(num) + ".txt", "w")
    currUserData = {}
    i = 0
    tmpCosine = 0
    userNumerator = 0
    userDenominator = 0
    while(i < len(testData)):
        tmpCosine = 0
        similarities = {}
        if(testData[i][2] == 0):
            currUser = testData[i][0]
            print("Numerator: " + str(userNumerator) + " ---- Denominator: " + str(userDenominator))
            currUserAverage = userNumerator/userDenominator
            print(currUserAverage)
            for k in range(i, len(testData)):
                if(testData[i][0] != currUser):
                    break
                else:
                    movie = int(testData[k][1]-1)
                    tmpCosine = 0
                    similarities = {}
                    numerator = 0
                    denominator = 0
                    for key, value in currUserData.items(): #Get all the sims
                        tmpCosine = cosine_similarity(trainData[movie], trainData[key], currUserAverage)
                        if(tmpCosine != 0):
                            similarities[key] = tmpCosine
                    for key, value in similarities.items(): #Calculate the weighted similarity
                            numerator += value*currUserAverage
                            denominator += value
                    if(denominator != 0):
                        rating = round(numerator/denominator)
                    if(rating == 0):
                        t = str(int(testData[movie][0])) + " " + str(int(testData[movie][1])) + " 3\n"
                        f.write(t)
                    else:
                        t = str(int(testData[movie][0])) + " " + str(int(testData[movie][1])) + " " + str(int(rating)) + "\n"
                        f.write(t)
                    i += 1
                currUserData = {}
                userNumerator = 0
                userDenominator = 0
        else:
            currUserData[int(testData[i][1]-1)] = int(testData[i][2])
            userNumerator += testData[i][2]
            userDenominator += 1
            i += 1

if __name__ == '__main__':
    if(sys.argv[1] == "5"):
        runItemBased(5)
    elif(sys.argv[1] == "10"):
        runItemBased(10)
    elif(sys.argv[1] == "20"):
        runItemBased(20)
    else:
        print("Invalid System Arguement. Please run script again")
        exit()