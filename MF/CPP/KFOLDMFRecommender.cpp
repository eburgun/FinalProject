


#include "KFOLDMFRecommender.h"

KFOLDMFRecommender::KFOLDMFRecommender(int kValue, double lambda, double epsilon, int maxIter):
    kVal(kValue), lambdaVal(lambda), epsVal(epsilon),iterations(maxIter)
{
    srand(time(NULL));
}


void KFOLDMFRecommender::createPandQ(CSR * trainingSet)
{
    pMatrix = new double*[trainingSet->rows];
    qMatrix = new double*[trainingSet->columns];
    for(int i = 0; i < trainingSet->rows; i++){
        pMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++){

            pMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }

    }
    for(int i = 0; i < trainingSet->columns; i++){
        qMatrix[i] = new double[kVal];
        for(int j = 0; j< kVal; j++){

            qMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void KFOLDMFRecommender::cleanUpPandQ(CSR * trainingSet)
{
    for(int i = 0; i<trainingSet->rows; i++){
        delete [] pMatrix[i];
        pMatrix[i] = NULL;

    }
    for(int i = 0; i<trainingSet->columns; i++){

        delete [] qMatrix[i];
        qMatrix[i] = NULL;

    }
    delete [] pMatrix;

    delete [] qMatrix;

}

void KFOLDMFRecommender::changeKValue(int newK, CSR * trainingSet)
{
    kVal = newK;
    cleanUpPandQ(trainingSet);
    createPandQ(trainingSet);
}

void KFOLDMFRecommender::changeLambda(double newLambda)
{
    lambdaVal = newLambda;
}

double KFOLDMFRecommender::fFunction(CSR * trainingSet)
{
    double pNorm = fNorm(pMatrix, trainingSet->rows);
    double qNorm = fNorm(qMatrix, trainingSet->columns);

    double lambdaQuantity = (pNorm + qNorm) * lambdaVal;
    double fSum = 0.0;
    for(int i = 0; i < trainingSet->rows;i++)
    {
        for(int j = 0; j <  trainingSet->columns;j++)
        {
            fSum += pow((trainingSet->getElement(i,j) - funcDotProduct(pMatrix[i],qMatrix[j])),2);
        }
    }
    return fSum + lambdaQuantity;
}

double KFOLDMFRecommender::funcDotProduct(double * a, double * b)
{
    double product = 0.0;
    for(int i = 0; i < kVal; i++)
    {
        product += a[i]*b[i];
    }
    return product;
}

double KFOLDMFRecommender::fNorm(double ** matrix,int dimension)
{
    double norm = 0.0;
    for(int i = 0; i <  dimension; i++){
        for(int j = 0; j < kVal; j++){
            norm += pow(matrix[i][j],2);
        }
    }
    return norm;
}

void KFOLDMFRecommender::LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate,std::string matrixId)
{
    double lambdaValue = 1 - (lambdaVal * 2 * learningRate);
    for(int i = 0; i < dataSet->rows;i++){
        double newItem[kVal];
        double sumMatrix[kVal];
        for(int j = dataSet->rowPtr[i]; j < dataSet->rowPtr[i+1]; j++){
            double dotProduct = 0.0;
            if(matrixId == "p"){
                dotProduct = funcDotProduct(solvingMatrix[i],fixedMatrix[dataSet->columnIndex[j]]);
            }
            else if(matrixId == "q"){
                dotProduct = funcDotProduct(fixedMatrix[dataSet->columnIndex[j]],solvingMatrix[i]);
            }
            double sumMult = (dataSet->ratingVals[j] - dotProduct);
            for(int k = 0; k < kVal; k++){
                sumMatrix[k] += fixedMatrix[dataSet->columnIndex[j]][k] * sumMult;
            }
        }

        for(int j = 0; j < kVal; j++){
            newItem[j] = solvingMatrix[i][j] * lambdaValue;
            sumMatrix[j] = sumMatrix[j] * (2 * learningRate);
            solvingMatrix[i][j] = sumMatrix[j] + newItem[j];
        }
    }
}

void KFOLDMFRecommender::trainSystem(CSR * trainingSet, CSR * transposeSet)
{
    int i = 0;
    double lastIter = 0.0;

    while(i <  iterations){

        LS_GD(trainingSet, qMatrix, pMatrix, 0.0005, "p");

        LS_GD(transposeSet, pMatrix, qMatrix, 0.0005, "q");

        double curIter = fFunction(trainingSet);

        if(i > 0 && sqrt(pow((curIter - lastIter),2)/lastIter) < epsVal){
            break;
        } else {
          lastIter = curIter;
        }
        i++;

    }


}

double KFOLDMFRecommender::mSE(CSR * testingSet)
{
    double mse = 0.0;
    for(int i = 0; i < testingSet->rows;i++){
        for(int j = testingSet->rowPtr[i]; j < testingSet->rowPtr[i+1]; j++){
            double prediction = funcDotProduct(pMatrix[i],qMatrix[testingSet->columnIndex[j]]);
            mse += pow(testingSet->ratingVals[j] - prediction,2);
            std::cout << prediction << std::endl;
        }
    }

    mse /= testingSet->nonZeroValues;
    return mse;
}

double KFOLDMFRecommender::rMSE(double mse)
{
    double rmse = sqrt(mse);
    std::cout << "k = " + std::to_string(kVal) + " lambda = " + std::to_string(lambdaVal) + " maxIters = " + std::to_string(iterations) + " epsilon = " + std::to_string(epsVal) + " mse = " + std::to_string(mse) + " rmse = " + std::to_string(rmse) << std::endl;
    return rmse;
}

void KFOLDMFRecommender::kFoldsTest(std::string trainStart, std::string testStart, std::string coldStart)
{
    std::string outPutFile = "kFoldsResults";
    for(int i = 1; i < 6; i ++)
    {
        std::ofstream outfile (outPutFile + std::to_string(i) + ".txt");
        time_t kFoldTestingStart = clock();
        CSR * trainingSet = new CSR(trainStart + std::to_string(i) +".txt");
        CSR * transposeSet = new CSR(trainStart + std::to_string(i) + ".txt");
        transposeSet->transpose();
        CSR * testingSet = new CSR(testStart + std::to_string(i) + ".txt");
        CSR * coldSet = new CSR(coldStart + std::to_string(i) + ".txt");

        testingMethod(trainingSet,transposeSet,testingSet,coldSet,outfile);

        delete trainingSet;

        delete transposeSet;

        delete testingSet;

        delete coldSet;
        outfile << "Total Testing Time: ";
        outfile << (double)(clock() - kFoldTestingStart) * 1000.0/CLOCKS_PER_SEC;
        outfile << "\n";
        outfile.close();
    }


}

double * KFOLDMFRecommender::createAverageUser(CSR * trainingSet)
{
    double * averageUser = new double[kVal];
    for(int i = 0; i < kVal; i++)
    {
        averageUser[i] = 0.0;
    }
    for(int i = 0; i < trainingSet->rows; i++)
    {
        for(int j = 0; j < kVal; j++)
        {
            averageUser[j] += pMatrix[i][j];
        }
    }
    for(int i = 0; i < kVal; i++)
    {
        averageUser[i] /= trainingSet->rows;
    }
    return averageUser;
}
void KFOLDMFRecommender::trainSingleUser(double * userMatrix, int usersItem, int usersRating)
{

    double learningRate = 0.00005;
    double lambdaValue = 1 - (lambdaVal * learningRate * 2);
    double userItem[kVal];
    double sumItemErrors[kVal];
    double userDotProduct = funcDotProduct(userMatrix,qMatrix[usersItem]);
    double userRatingError = (usersRating - userDotProduct);

    for(int i =0; i < kVal; i++)
    {
        sumItemErrors[i] = qMatrix[usersItem][i] * userRatingError;

        userItem[i] = userMatrix[i] * lambdaValue;

        sumItemErrors[i] = sumItemErrors[i] * (learningRate * 200);

        userMatrix[i] = sumItemErrors[i] + userItem[i];
    }

    double itemsUser[kVal];
    double sumUserErrors[kVal];
    double itemDotProduct = funcDotProduct(qMatrix[usersItem],userMatrix);
    double itemRatingError = (usersRating - itemDotProduct);

    for(int i = 0; i < kVal; i++)
    {
        sumUserErrors[i] = userMatrix[i] * itemRatingError;
        itemsUser[i] = qMatrix[usersItem][i] * lambdaValue;
        sumUserErrors[i] = sumUserErrors[i] * (learningRate * 2);
        qMatrix[usersItem][i] = sumUserErrors[i] + itemsUser[i];
    }
}
void KFOLDMFRecommender::coldStartTesting(CSR * coldSet, double * averageUser, CSR * trainingSet, std::ofstream& outfile)
{
    time_t coldTrainStart = clock();
    double ** newUserMatrix = new double*[coldSet->rows];
    double totalHR = 0.0;
    for(int i = 0; i < coldSet->rows; i++)
    {
        newUserMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++)
        {
            newUserMatrix[i][j] = averageUser[j];
        }
    }
    double setAverageHR = 0.0;
    double setAverageARHR = 0.0;
    double setAverageMSE = 0.0;
    double setAverageRMSE = 0.0;
    for(int i = 0; i < coldSet->rows; i++)
    {
        outfile << "Cold Start User #";
        outfile << std::to_string(i);
        outfile << ": \n";
        int userItemCount = coldSet->rowPtr[i+1] - coldSet->rowPtr[i];
        int ** untrainedItem = new int*[userItemCount];
        int * untrainedRatings = new int[userItemCount];
        int trainedItems = 0;
        for(int j = 0; j < userItemCount; j++)
        {
            untrainedItem[j] = new int[2];
            untrainedItem[j][0] = coldSet->columnIndex[j+coldSet->rowPtr[i]];
            untrainedItem[j][1] = 0;
            untrainedRatings[j] = coldSet->ratingVals[j+coldSet->rowPtr[i]];
        }
        double userAverageHR = 0.0;
        double userAverageARHR = 0.0;
        double userAverageMSE = 0.0;
        double userAverageRMSE = 0.0;
        while(trainedItems <  userItemCount)
        {


            double ** userRecs = predictUserRecs(newUserMatrix[i], coldSet->columns, untrainedItem, userItemCount, trainingSet);

            double hits = 0;
            double arHits = 0;
            double cumulativeError = 0.0;
            for(int i = 0; i < userItemCount; i++)
            {
                for(int j = 0; j < 20; j++)
                {
                    if(untrainedItem[i][0] == userRecs[j][1] && untrainedItem[i][1] != -1)
                    {
                        hits++;
                        arHits += 1.0/(j+1);
                        cumulativeError += pow(untrainedRatings[i] - userRecs[j][0],2);
                    }
                }
            }
            float potentialHR = 0.0;
            float potentialARHR = 0.0;
            double mse = 0.0;
            if(userItemCount - trainedItems < 20.0 && hits != 0)
            {
                potentialHR = hits / (userItemCount-trainedItems);
                potentialARHR = arHits / (userItemCount-trainedItems);
                mse = cumulativeError / hits;
            }
            else if(hits != 0)
            {
                potentialHR = hits/20.0;
                potentialARHR = arHits/20.0;
                mse = cumulativeError/hits;
            }
            double rmse = sqrt(mse);
            outfile << "Potential HR: ";
            outfile << potentialHR;
            outfile << " Potential ARHR: ";
            outfile << potentialARHR;
            outfile << " Hits MSE: ";
            outfile << mse;
            outfile << " Hits RMSE: ";
            outfile << rmse;
            outfile << "\n";

            trainedItems++;
            int randomValue = rand() % userItemCount;
            while(untrainedItem[randomValue][1] == -1)
            {
                randomValue = rand() % userItemCount;
            }
            int usersNext = untrainedItem[randomValue][0];
            untrainedItem[randomValue][1] = -1;
            trainSingleUser(newUserMatrix[i], usersNext, untrainedRatings[randomValue]);

            for(int j = 0; j < 20; j ++)
            {
                delete[] userRecs[j];
                userRecs[j] = NULL;
            }
            delete [] userRecs;
            userAverageHR += potentialHR;
            userAverageARHR += potentialARHR;
            userAverageMSE += mse;
            userAverageRMSE += rmse;
        }
        userAverageHR /= userItemCount;
        userAverageARHR /= userItemCount;
        userAverageMSE /= userItemCount;
        userAverageRMSE /= userItemCount;
        outfile << "User Average Metrics: HR: ";
        outfile << userAverageHR;
        outfile << " ARHR: ";
        outfile << userAverageARHR;
        outfile << " MSE: ";
        outfile << userAverageMSE;
        outfile << " RMSE: ";
        outfile << userAverageRMSE;
        outfile << "\n";
        outfile << "\n";
        for(int i = 0; i < userItemCount; i ++)
        {
            delete [] untrainedItem[i];
            untrainedItem[i] = NULL;
        }
        delete [] untrainedItem;
        delete [] untrainedRatings;
        setAverageHR += userAverageHR;
        setAverageARHR += userAverageARHR;
        setAverageMSE += userAverageMSE;
        setAverageRMSE += userAverageRMSE;
    }

    for(int i = 0; i < coldSet->rows; i++)
    {
        delete [] newUserMatrix[i];
    }
    delete [] newUserMatrix;
    setAverageHR /= coldSet->rows;
    setAverageARHR /= coldSet->rows;
    setAverageMSE /= coldSet->rows;
    setAverageRMSE /= coldSet->rows;
    outfile << "Set Average Metrics: HR: ";
    outfile << setAverageHR;
    outfile << " ARHR: ";
    outfile << setAverageARHR;
    outfile << " MSE: ";
    outfile << setAverageMSE;
    outfile << " RMSE: ";
    outfile << setAverageRMSE;
    outfile << "\n";
    outfile << "ColdStart Training Time: ";
    outfile << (double)(clock() - coldTrainStart) * 1000.0/CLOCKS_PER_SEC;
    outfile << "\n";
    outfile << "\n";


}

double ** KFOLDMFRecommender::predictUserRecs(double * user, int nItems, int ** trained, int sizeOfTrained, CSR * trainingSet)
{
    int n = 20;
    double ** recommendations = new double*[n];
    for(int i = 0; i < n; i++)
    {
        recommendations[i] = new double[2];
        recommendations[i][0] = 0;
        recommendations[i][1] = -1;
    }

    double ** userBase = new double*[trainingSet->rows];
    for(int i = 0; i < trainingSet->rows; i++)
    {
        userBase[i] = new double[2];
        userBase[i][0] = cosSimil(pMatrix[i], user);
        userBase[i][1] = i;
    }

    quickSort(userBase,trainingSet->rows,0);
    int similarUsers[kVal];
    for(int i = 0; i < kVal; i++)
    {
        similarUsers[i] = userBase[i][1];
    }
    double ** ratingPredictions = new double*[trainingSet->columns];
    for(int i = 0; i < trainingSet->columns; i++)
    {
        ratingPredictions[i] = new double[2];
        ratingPredictions[i][0] = 0.0;
        ratingPredictions[i][1] = i;
    }
    for(int i = 0; i < kVal; i++)
    {
        for(int j = trainingSet->rowPtr[similarUsers[i]]; j < trainingSet->rowPtr[similarUsers[i]+1];j++)
        {
            ratingPredictions[trainingSet->columnIndex[j]][0]++;
        }
    }
    quickSort(ratingPredictions,trainingSet->columns,0);
    int x = 0;
    int inRecs = 0;
    while(x < trainingSet->columns && inRecs < n)
    {
        bool inTrained = false;
        for(int i = 0; i < sizeOfTrained; i++)
        {
            if(ratingPredictions[x][1] == trained[i][0] && trained[i][1] == -1)
            {
                inTrained = true;
            }
        }
        if(!inTrained)
        {
            recommendations[inRecs][0] = funcDotProduct(user,qMatrix[int(ratingPredictions[x][1])]);
            recommendations[inRecs][1] = ratingPredictions[x][1];
            inRecs++;
        }
        x++;
    }
    quickSort(recommendations, 20, 0);
    for(int i = 0; i < trainingSet->columns; i++)
    {
        delete [] ratingPredictions[i];
    }
    delete [] ratingPredictions;
    for(int i = 0; i < trainingSet->rows; i++)
    {
        delete [] userBase[i];
    }
    delete [] userBase;

    return recommendations;
}
double KFOLDMFRecommender::cosSimil(double * a, double * b)
{
    double numerator = funcDotProduct(a,b);
    double lengthA = sqrt(funcDotProduct(a,a));
    double lengthB = sqrt(funcDotProduct(b,b));
    return numerator/(lengthA * lengthB);
}
void KFOLDMFRecommender::quickSort(double ** sArray, int arraySize,int index)
{
    part(sArray, 0, arraySize - 1,index);
}

void KFOLDMFRecommender::part(double ** sArray, int left, int right,int index)
{
    if((right - left) > 5)
    {
        double pivotValue = pivot(sArray, left, right,index);

        int i = left + 1;
        int j = right;

        while (i < j)
        {
            while(pivotValue > sArray[--j][index]){}
            while (sArray[++i][index] > pivotValue){}
            if(i < j)
            {
                swapper(sArray,i,j);
            }
        }
        swapper(sArray,left + 1, j);

        part(sArray,left,j-1,index);
        part(sArray,j+1,right,index);
    }
    else
    {
        insertionSort(sArray,left,right,index);
    }

}
double KFOLDMFRecommender::pivot(double ** sArray,int left,int right,int index)
{
    int center = (left + right) / 2;

    if (sArray[center][index] < sArray[right][index])
    {
        swapper(sArray,right,center);
    }
    if (sArray[left][index] < sArray[right][index])
    {
        swapper(sArray,left,right);
    }
    if (sArray[left][index] < sArray[center][index])
    {
        swapper(sArray,left,center);
    }

    swapper(sArray,center,(left + 1));
    return sArray[left + 1][index];
}

void KFOLDMFRecommender::swapper(double ** sArray, int a, int b)
{
    double * temp = sArray[a];
    sArray[a] = sArray[b];
    sArray[b] = temp;
}

void KFOLDMFRecommender::insertionSort(double ** sArray, int left, int right, int index)
{
    int i;
    int j;
    for(i = left; i <= right; i ++)
    {
        double * tmp = sArray[i];
        j = i + left;
        for( j = i; j > left && tmp[index] > sArray[j-1][index]; j--)
        {
            sArray[j] = sArray[j-1];
        }
        sArray[j] = tmp;
    }
}

void KFOLDMFRecommender::testingMethod(CSR * trainingSet, CSR * transposeSet, CSR * testingSet, CSR * coldSet, std::ofstream& outfile)
{
    int kVals [] = {10,50};
    float lambVals[] = {0.01,0.1,1,10};
    int iters [] = {50,100,200};
    float epsilonVals [] = {0.0001, 0.001, 0.01};
    for(int i = 0; i < 2; i++)
    {
        kVal = kVals[i];
        for(int j = 0; j < 4; j++)
        {
            lambdaVal = lambVals[j];
            for(int k = 0; k < 3; k++)
            {
                iterations = iters[k];
                for(int l = 0; l < 3; l++)
                {
                    epsVal = epsilonVals[l];
                    createPandQ(trainingSet);
                    clock_t trainStart = clock();
                    trainSystem(trainingSet, transposeSet);
                    clock_t trainFinish = clock();
                    clock_t testStart = clock();
                    float mse = mSE(testingSet);
                    float rmse = rMSE(mse);
                    clock_t testFinish = clock();
                    outfile << kVal;
                    outfile << " ";
                    outfile << lambdaVal;
                    outfile << " ";
                    outfile << iterations;
                    outfile << " ";
                    outfile << epsVal;
                    outfile << " ";
                    outfile << mse;
                    outfile << " ";
                    outfile << rmse;
                    outfile << " ";
                    outfile << (double)(trainFinish - trainStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << " ";
                    outfile << (double)(testFinish - testStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << "\n";
                    double * averageUser = createAverageUser(trainingSet);

                    coldStartTesting(coldSet, averageUser,trainingSet, outfile);
                    cleanUpPandQ(trainingSet);
                }
            }
        }
    }
}
