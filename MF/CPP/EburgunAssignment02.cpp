//
//  EburgunAssignment02.cpp
//
//
//  Created by Evan Burgun on 10/3/16.
//
//

#include <stdio.h>
#include "KFOLDMFRecommender.h"

int main(){

    bool running = true;
    std::string outPut = "Output.txt";
    std::string trainingFile = "../../Datasets/OriginalDataSets/SmallTrainSet.txt";
    std::string testFile = "../../Datasets/OriginalDataSets/SmallTestSet.txt";
    std::string kFoldTrain = "../../Datasets/TrainingSet/SmallTrainingSet";
    std::string kFoldTest = "../../Datasets/TestingSet/SmallTestingSet";
    std::string kFoldColdStart = "../../Datasets/ColdTests/SmallColdStart";
    int kVal = 10;
    float lambdaVal = 0.1;
    float epsilon = 0.001;
    int maxTries = 200;

    KFOLDMFRecommender * rec = new KFOLDMFRecommender(kVal,lambdaVal,epsilon,maxTries);



    std::cout << "Hello, welcome to my Recommender System!" << std::endl;
    std::cout << "Please wait while we load your data." << std::endl;

    std::cout << "Data Loaded" << std::endl;
    std::cout << "Please choose from the options below:" << std::endl;
    std::cout << "1. Define K Value. (Default == 10)(Not implemented)" << std::endl;
    std::cout << "2. Define Lambda Value. (Default == 0.1)" << std::endl;
    std::cout << "3. Train System(Not Implemented)" << std::endl;
    std::cout << "4. Test trained System(Not Implemented)" << std::endl;
    std::cout << "5. Create Recommendations and save them(Not implemented)" << std::endl;
    std::cout << "6. Test Cold Start(Not Implemented)" << std::endl;
    std::cout << "7. Create Test Report(Not implemented)" << std::endl;
    std::cout << "8. Run Kfolds testing" << std::endl;
    std::cout << "Q. Exit" << std::endl;

    bool recHasRun = false;
    while(running){
        std::cout << "Which option would you like to choose? " << std::endl;
        std::string userInput;
        std::cin >> userInput;
        if(userInput == "1"){
            std::string newKVal;
            std::cin >> newKVal;
            try
            {
                int kValue = atoi(newKVal.c_str());
                
            }
            catch(int e)
            {
                std::cout << "An error has occured" <<  std::endl;
            }
            userInput = "0";
            newKVal = "0";
        }else if(userInput == "2"){
            std::string newLambda;
            std::cin >> newLambda;
            try
            {
                float lambda = strtof(newLambda.c_str(),NULL);
                rec->changeLambda(lambda);
            }
            catch(int e)
            {
                std::cout << "An error has occured" << std::endl;
            }
            userInput = "0";
        }else if(userInput == "8"){
            rec->kFoldsTest(kFoldTrain, kFoldTest, kFoldColdStart);
            userInput = "0";
        }else if(userInput == "Q" || userInput == "q"){
            std::cout << "Have a nice day!" << std::endl;
            running = false;
        }else {
            std::cout << "I'm sorry, I didn't quite catch that." << std::endl;
        }

    }

    delete rec;

    return 0;

}
