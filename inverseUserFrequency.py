import sys
import numpy as np
import math

#Load training data into matrix
trainData = np.loadtxt("train.txt")

def computeIUF(index): #IUF function
    numUsers = 0
    for user in trainData:
        if(user[index] != 0):
            numUsers += 1
    
    IUF = math.log(200/numUsers)
    return IUF

def ratingAvg(arr): #Average rating function
    numerator = 0.0
    denominator = 0.0
    for rating in arr:
        if(rating != 0):
            numerator += rating
            denominator += 1
    
    if(denominator == 0):
        return 0
    else:
        return float(numerator/denominator)

def pearson_correlationW(train, test):
    assert(len(train) == len(test))

    trainAvg = ratingAvg(train)
    testAvg = ratingAvg(test)
    
    numerator = 0
    denominatorTest = 0
    denominatorTrain = 0
    for trainRating, testRating in zip(train, test):
        if(trainRating != 0 and testRating != 0):
            numerator += ((trainRating-trainAvg)*(testRating-testAvg))
            denominatorTrain += pow(trainRating-trainAvg, 2)
            denominatorTest += pow(testRating-testAvg, 2)
    
    denominator = float(math.sqrt(denominatorTrain))*float(math.sqrt(denominatorTest))
        
    if(denominator == 0):
        return 0.0
    else:
        return float(numerator/denominator)
    
def runPearson(num):
    testData = np.loadtxt("test" + str(num) + ".txt")
    f = open("results" + str(num) + ".txt", "w")
    currUserData = np.zeros(1000, dtype=np.int)
    i = 0
    while(i < len(testData)):
        if(testData[i][2] == 0):
            print(testData[i][0])
            correlations = {}
            tmpCorr = 0.0
            for j in range(0, len(trainData)):
                tmpCorr = pearson_correlationW(trainData[j], currUserData)
                if(tmpCorr != 0):
                    correlations[j] = tmpCorr
            currUser = testData[i][0]
            testAvg = ratingAvg(currUserData)
            for k in range(i, len(testData)):
                if(testData[k][0] != currUser):
                    break
                numerator = 0.0
                denominator = 0.0
                rating = 0.0
                idx = int(testData[k][1])-1
                for key, value in correlations.items():
                    if(trainData[key][idx] != 0):
                        numerator += ((trainData[key][idx]-ratingAvg(trainData[key]))*value)
                        denominator += abs(value)
                if(denominator != 0.0):
                    #print(computeIUF(idx))
                    rating = testAvg+float(numerator/denominator)*computeIUF(idx)
                    if(rating <= 0.5):
                        rating = 1
                    elif(rating > 5):
                        rating = 5
                else:
                    rating = testAvg
                t = str(int(testData[k][0])) + " " + str(int(testData[k][1])) + " " + str(int(round(rating))) + "\n"
                f.write(t)
                i += 1
            currUserData = np.zeros(1000, dtype=np.int)
        else:
            currUserData[int(testData[i][1]-1)] = int(testData[i][2]) 
            i += 1    
                                    
if __name__ == '__main__':
    if(sys.argv[1] == "5"):
        runPearson(5)
    elif(sys.argv[1] == "10"):
        runPearson(10)
    elif(sys.argv[1] == "20"):
        runPearson(20)
    else:
        print("Invalid System Arguement. Please run script again")
        exit()