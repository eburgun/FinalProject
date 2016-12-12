import sys
import CSR
import random
import time

setSize = sys.argv[1]
random.seed(time.clock())
trainingFile = "./OriginalDataSets/"
testingFile = "./OriginalDataSets/"

trainingOutput = "./TrainingSet/"
testingOutput = "./TestingSet/"
coldStartOutput = "./ColdStart/"

if(setSize == "1"):
    trainingFile += "LargeTrainSet.txt"
    testingFile += "LargeTestSet.txt"
    trainingOutput += "LargeTrainingSet"
    testingOutput += "LargeTestingSet"
    coldStartOutput += "LargeColdStart"
else:
    trainingFile += "SmallTrainSet.txt"
    testingFile += "SmallTestSet.txt"
    trainingOutput += "SmallTrainingSet"
    testingOutput += "SmallTestingSet"
    coldStartOutput += "SmallColdStart"

trainingData = CSR.CSR(trainingFile)
testingData = CSR.CSR(testingFile)
for x in xrange(5):
    randomUsers = []
    usedUsers = [0] * trainingData.rows
    i = 0
    randInt = random.randint(0,trainingData.rows - 1)
    while (i < trainingData.rows/5):
        if(usedUsers[randInt] == 1):
            randInt = random.randint(0, trainingData.rows - 1)
        else:
            usedUsers[randInt] = 1
            randomUsers.append(randInt)
            i += 1

    initialUserSize = trainingData.rows - len(randomUsers)
    randomUserEntries = 0
    for i in xrange(len(randomUsers)):
        randomUserEntries += trainingData.row_ptr[randomUsers[i]+1] - trainingData.row_ptr[randomUsers[i]]
    initialUserData = trainingData.nonzero_values - randomUserEntries

    newTrainingFile = open(trainingOutput+str(x + 1)+".txt","w")
    newTestFile = open(testingOutput+str(x + 1)+".txt","w")
    coldStartFile = open(coldStartOutput+str(x + 1)+".txt","w")

    newTrainingFile.write(str(initialUserSize) + " " + str(trainingData.columns) + " " + str(initialUserData) + "\n")
    newTestFile.write(str(initialUserSize) + " " + str(trainingData.columns) + " " + str(initialUserSize) + "\n")
    coldStartFile.write(str(len(randomUsers)) + " " + str(trainingData.columns) + " " + str(randomUserEntries) + "\n")

    for i in xrange(trainingData.rows):
        userString = ""
        for j in xrange(trainingData.row_ptr[i],trainingData.row_ptr[i + 1]):
            userString += str(trainingData.column_idx[j] + 1) + " " + str(trainingData.rating[j]) + " "
        if i in randomUsers:
            coldStartFile.write(userString + str(testingData.column_idx[i] + 1) + " " + str(testingData.rating[i]) + "\n" )
        else:
            newTrainingFile.write(userString + "\n")
            newTestFile.write(str(testingData.column_idx[i] + 1) + " " + str(testingData.rating[i]) + "\n")
    del randomUsers[:]
    newTrainingFile.close()
    newTestFile.close()
    coldStartFile.close()

