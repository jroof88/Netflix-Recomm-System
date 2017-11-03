import sys
import numpy as np
import math
import time

#Load training data into matrix
trainData = np.loadtxt("train.txt")
trainDataTranspose = np.transpose(trainData)

########## COSINE SIMILARITY AND ITS FUNCTIONS ##########

def cosine_similarity(currUserData, loc, testData):
    similarities = {}
    tmpCosine = 0
    rating = 0
    for j in range(0, len(trainData)):
        tmpCosine = cosineSim(trainData[j], currUserData)
        if(tmpCosine != 0):
            similarities[j] = tmpCosine
    numerator = 0
    denominator = 0
    idx = int(testData[loc][1])-1
    rating = 0
    for key, value in similarities.items():
        if(trainData[key][idx] != 0):
            print(trainData[key][idx])
            numerator += value*trainData[key][idx]
            denominator += value
    if(denominator != 0):
        rating = round(numerator/denominator)
    if(rating == 0):
        rating = 3
    
    return rating
        
def cosineSim(train, test):
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

########## PEARSON CORRELATION AND ITS FUNCTIONS ##########

def pearson_similarity(currUserData, loc, testData):
    correlations = {}
    tmpCorr = 0.0
    rating = 0
    for j in range(0, len(trainData)):
        tmpCorr = pearson_correlationW(trainData[j], currUserData)
        if(tmpCorr != 0):
            correlations[j] = tmpCorr
    testAvg = ratingAvg(currUserData)
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    idx = int(testData[loc][1])-1
    for key, value in correlations.items():
        if(trainData[key][idx] != 0):
            numerator += ((trainData[key][idx]-ratingAvg(trainData[key]))*value)
            denominator += abs(value)
    if(denominator != 0.0):
        rating = testAvg+float(numerator/denominator)
    if(rating <= 0.5):
        rating = 1
    elif(rating > 5):
        rating = 5
    else:
        rating = testAvg
    
    return rating
    
def ratingAvg(arr):
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

########## PEARSON CORRELATION WITH IUF (USES SOME FUNCTION FROM REGULAR PEARSON) ##########

def pearson_similarity_IUF(currUserData, loc, testData):
    correlations = {}
    tmpCorr = 0.0
    for j in range(0, len(trainData)):
        tmpCorr = pearson_correlationW(trainData[j], currUserData)
        if(tmpCorr != 0):
            correlations[j] = tmpCorr
    testAvg = ratingAvg(currUserData)
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    idx = int(testData[loc][1])-1
    for key, value in correlations.items():
        if(trainData[key][idx] != 0):
            numerator += ((trainData[key][idx]-ratingAvg(trainData[key]))*value)
            denominator += abs(value)
    if(denominator != 0.0):
        rating = testAvg+float(numerator/denominator)*computeIUF(idx)
        if(rating <= 0.5):
            rating = 1
        elif(rating > 5):
            rating = 5
    else:
        rating = testAvg
    
    return rating

def computeIUF(index):
    numUsers = 0
    for user in trainData:
        if(user[index] != 0):
            numUsers += 1
    
    IUF = math.log(200/numUsers)
    return IUF
    
########## PEARSON CORRELATION WITH CASE MODIFICATION WITH ITS FUNCTIONS (USES SOME FUNCTIONS FROM REGULAR PEARSON) ##########
def pearson_similarity_CaseMod(currUserData, loc, testData):
    correlations = {}
    tmpCorr = 0.0
    for j in range(0, len(trainData)):
        tmpCorr = pearson_correlationW(trainData[j], currUserData)
        if(tmpCorr != 0):
            correlations[j] = tmpCorr
    testAvg = ratingAvg(currUserData)
    numerator = 0.0
    denominator = 0.0
    rating = 0.0
    idx = int(testData[loc][1])-1
    for key, value in correlations.items():
        if(trainData[key][idx] != 0):
            numerator += ((trainData[key][idx]-ratingAvg(trainData[key]))*case_modification(abs(value)))
            denominator += case_modification(abs(value))

    if(denominator != 0.0):
        rating = testAvg+float(numerator/denominator)
        if(rating <= 0.5):
            rating = 1
        elif(rating > 5):
            rating = 5
    else:
        rating = testAvg

    return rating
    
def case_modification(val):
    newVal = pow(val, 2.5)
    return newVal    

def run(num):
    testData = np.loadtxt("test" + str(num) + ".txt")
    f = open("results" + str(num) + ".txt", "w")
    i = 0
    currUserData = np.zeros(1000, dtype=np.int)
    while(i < len(testData)):
        if(testData[i][2] == 0):
            cosineRating = cosine_similarity(currUserData, i, testData)
            pearsonRating = pearson_similarity(currUserData, i, testData)
            pearsonIUFRating = pearson_similarity_IUF(currUserData, i, testData)
            pearsonCaseModRating = pearson_similarity_CaseMod(currUserData, i, testData)
            rating = (cosineRating+pearsonRating+pearsonIUFRating+pearsonCaseModRating)/4
            t = str(int(testData[i][0])) + " " + str(int(testData[i][1])) + " " + str(int(round(rating))) + "\n"
            f.write(t)
            i += 1
        else:    
            currUserData[int(testData[i][1]-1)] = int(testData[i][2]) 
            i += 1
            
    
if __name__ == '__main__':
    start = time.time()
    if(sys.argv[1] == "5"):
        run(5)
    elif(sys.argv[1] == "10"):
        run(10)
    elif(sys.argv[1] == "20"):
        run(20)
    else:
        print("Invalid System Arguement. Please run script again")
        exit()