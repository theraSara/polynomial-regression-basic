#define SensorPIN A0
#define HumanYPIN A1
#define ControlPIN 10
#define BeginTrainingPIN A2
#define TestingModePB 6
#define ModelValidPB 5
#define TestPB 4

enum{Idle, TrainingStart, CollectingDataOnEntry, 
 CollectingDataOnStay, FitModel, CollectX, 
 CalculateY, UpdateControlOnEntry, UpdateControlOnStay,
 CollectTestData, EstimateY, AccumError, ShowError};

int current = Idle;

int previousCollectPB = 0, currentCollectPB = 1;
int previousBeginTrainingPB = 0, currentBeginTrainingPB = 1;
int previousTestingModePB = 0, currentTestingModePB = 1;
int previousModelValidPB = 0, currentModelValidPB = 1;
int previousTestPB = 0, currentTestPB = 1;

#define NR_END 1
#define N (10)
#define M (3)

// declaration of data storage for inputs and outputs 
mtx_type X[M][N] = {{1,1,1,1,1,1,1,1,1,1},
            {0,1,2,3,4,5,6,7,8,9},
            {1,1,1,1,1,1,1,1,1,1}};

#define TestingPortion 0.3
#define TrainingSize (int)((1-TestingPortion)*N)
#define TestingSize (int)(N-TrainingSize)

mtx_type Xtrain[M][TrainingSize];
mtx_type Xtest[M][TestingSize];

int RandomIndex[] = {3,7,2,6,8,9,5,1,4,0};

mtx_type XT[TrainingSize][M];
mtx_type XTXinv[M][M];
mtx_type XTXinvIntoX[M][TrainingSize];
mtx_type y[N] = {76,76,159,189,201,155,491,362,703,933};
mtx_type ytrain[TrainingSize];
mtx_type ytest[TestingSize];
mtx_type w[M];

bool DebugAIMode = true;

//Model Fitting Parameters
double sum,sumofsquares;
double detrecp;
double singleX; 
double singleY, Ytrue, Ypredicted;
double error, errorSquared, sumSquaredError, meanSquaredError;
double MSE, RMSE, MAE;
int errorPointsCount = 1; 
int mapped;
double sse = 0, mae = 0;
float stv = 0, mtv = 0;
float rSquared;
int Count = 0;
long previousMillis = 0;

MatrixMath Matrix;

bool delayG(int time)
{
    return (millis() - previousMillis) > time;
}

void PrintModelFittingMatrix()
{
    Matrix.Print((mtx_type*)X,M,N,"X");
    Matrix.Print((mtx_type*)y,1,N,"y"); 
}

void QuadraticModelFitting()
{
    Matrix.Transpose((mtx_type*)Xtrain,M,TrainingSize,(mtx_type*)XT);
    Matrix.Multiply((mtx_type*)Xtrain,(mtx_type*)XT,M,TrainingSize,
                M,(mtx_type*)XTXinv);
    Matrix.Invert((mtx_type*)XTXinv,M);
    Matrix.Multiply((mtx_type*)XTXinv,(mtx_type*)Xtrain,M,M,TrainingSize,
                (mtx_type*)XTXinvIntoX);
    Matrix.Multiply((mtx_type*)XTXinvIntoX,(mtx_type*)ytrain,M,TrainingSize,
                1,(mtx_type*)w);
    if(DebugAIMode) PrintModelFittingMatrix();
    Matrix.Print((mtx_type*)w,M,1,"w");
    double sse = 0, mae = 0; 
    for (int i = 0; i < TestingSize; i++)
    {
        sse += pow((w[2] * Xtest[2][i] + w[1] * Xtest[1][i] + w[0]) - ytest[i],
                                                                            2);
        mae += abs((w[2] * Xtest[2][i] + w[1] * Xtest[1][i] + w[0]) - ytest[i]);
    }
    MSE = sse/TestingSize;
    RMSE = sqrt(MSE);
    MAE = mae / TestingSize;
}

void performanceReport()
{
    if(DebugAIMode) {
        Serial.print("Mean Squared Error (MSE): ");
        Serial.println(MSE);
        Serial.print("Root Mean Squared Error (RMSE): ");
        Serial.println(RMSE);
        Serial.print("Mean Absolute Error (MAE): ");
        Serial.println(MAE);
    }
}

void train_test_split()
{
    for (int i = 0; i < TrainingSize; i++)
    {
        Xtrain[0][i] = 1;  
        Xtrain[1][i] = X[1][RandomIndex[i]];
        Xtrain[2][i] = X[2][RandomIndex[i]];
        ytrain[i] = y[RandomIndex[i]]; 
    }
    for (int i = 0; i < TestingSize; i++)
    {
        Xtest[0][i] = 1;
        Xtest[1][i] = X[1][RandomIndex[TrainingSize+i]];
        Xtest[2][i] = X[2][RandomIndex[TrainingSize+i]];
        ytest[i] = y[RandomIndex[TrainingSize+i]];
    }
}

void setup()
{
    Serial.begin(9600);
    if(DebugAIMode)
    {
        for(int i = 0; i < N; i++)
          X[2][i] = X[1][i]*X[1][i]; 
    }
    if(DebugAIMode) 
    {
        train_test_split();
        QuadraticModelFitting(); 
    }
    pinMode(BeginTrainingPIN, INPUT_PULLUP); 
    pinMode(TestingModePB,INPUT_PULLUP); 
    pinMode(ModelValidPB,INPUT_PULLUP); 
    pinMode(TestPB,INPUT_PULLUP); 
}

void loop()
{
    currentBeginTrainingPB = digitalRead(BeginTrainingPIN);
    currentTestingModePB = digitalRead(TestingModePB);
    currentModelValidPB = digitalRead(ModelValidPB);
    currentTestPB = digitalRead(TestPB);
    switch(current)
    {
    case Idle:    
        Serial.println("System Starting");
        if (currentBeginTrainingPB == 0 && previousBeginTrainingPB == 1)
          current = TrainingStart;
        if (currentTestingModePB == 0 && previousTestingModePB == 1)
          current = UpdateControlOnEntry;
    break;
    
    case TrainingStart:    
        Serial.print("Enter sensor value for training: ");
        while (!Serial.available()) {}  // Wait for user input
        X[1][Count] = Serial.parseInt();
        Serial.print(" ");
        Serial.println(X[1][Count]);
        
        Serial.print("Enter control value for training: ");
        while (!Serial.available()) {} 
        X[2][Count] = X[1][Count] * X[1][Count];
        y[Count] = Serial.parseInt();
        Serial.print(" ");
        Serial.println(y[Count]);
        Count++;
        Serial.print("Data Collected So Far: ");
        Serial.println(Count);
        if(Count == N) 
        {
          Count = 0;
          train_test_split();
          current = FitModel;
        }
    break;
    
    case CollectingDataOnEntry:
        Count++;	
        Serial.print("Data Collected So Far: ");
        Serial.println(Count); 
        current = CollectingDataOnStay; 
    break;
    
    case CollectingDataOnStay:
        X[1][Count-1] = analogRead(SensorPIN);
        X[2][Count-1] = X[1][Count-1]*X[1][Count-1]; 
        y[Count-1] = analogRead(HumanYPIN); 
        delay(100); 
        current = TrainingStart; 
    break;
    
    case FitModel:
        Serial.println("AI is Training"); 
        QuadraticModelFitting();
        performanceReport();
        current = CollectX;
    break;
    
    case CollectX:
        singleX = analogRead(SensorPIN);
        Serial.print("X = ");
        Serial.println(singleX);
        if(currentTestingModePB == 0 && previousTestingModePB == 1)
          current = CollectTestData; 
        if(currentBeginTrainingPB == 0 && previousBeginTrainingPB == 1)
          current = TrainingStart; 
        current = CalculateY; 
    break;
    
    case CalculateY:
        singleY = w[2]*singleX*singleX + w[1]*singleX + w[0]; 
        Serial.print("Y = ");
        Serial.println(singleY);
        current = UpdateControlOnEntry; 
    break;
    
    case UpdateControlOnEntry:
        previousMillis = millis(); 
        current = UpdateControlOnStay;
    break;
    
    case UpdateControlOnStay:
        mapped = map((int)singleY,(int)w[0],(int)(((1023*1023)*(w[2]))+
                                            1023*w[1]+w[0]),0,255); 
        analogWrite(ControlPIN,mapped);
        Serial.print("C = ");
        Serial.println(mapped);
        if(previousTestingModePB == 1 && currentTestingModePB == 0)
          errorPointsCount = 1; 
          sumSquaredError = 0;
          current =  CollectTestData;
          Serial.println("Testing Mode Started");
        if(delayG(1000))
          current = CollectX;
    break;
    
    case CollectTestData:  
        singleX = map(analogRead(SensorPIN),73,1002,0,9);
        Ytrue = analogRead(HumanYPIN);
        Serial.println("Enter Sensor (Temp) Value for Testing: ");
        while (!Serial.available()) {}
        singleX = Serial.parseInt();
        Serial.print(" ");
        Serial.println(singleX);
        
        Serial.println("Enter Control Value for Testing: ");
        while (!Serial.available()) {}  
        Ytrue = Serial.parseInt();
        Serial.print(" ");
        Serial.println(Ytrue);
        if(currentTestPB == 0 && previousTestPB == 1)	
            errorPointsCount++;
            current = EstimateY;
        
        if(previousModelValidPB == 1 && currentModelValidPB == 0)
           current = CollectX;
    break;
    
    case EstimateY:
        Ypredicted = w[2]*singleX*singleX + w[1]*singleX + w[0];
        Serial.print("Y Predicted: ");
        Serial.println(Ypredicted);
        current = AccumError;
    break;
    
    case AccumError:
        error = Ytrue - Ypredicted;
        errorSquared = error*error;
        sumSquaredError += errorSquared;
        meanSquaredError = sumSquaredError/errorPointsCount;
        RMSE = sqrt(meanSquaredError);
        sse += pow((w[2]*singleX*singleX+w[1]*singleX+w[0]) - Ytrue, 2);
        mae += fabs((w[2]*singleX*singleX+w[1]*singleX+w[0]) - Ytrue);
        mtv += (mtv *(errorPointsCount - 1) + Ytrue) / errorPointsCount;
        stv += pow(Ytrue - mtv, 2);
        rSquared = 1 - (sumSquaredError/stv);
        MSE = sse/errorPointsCount;
        RMSE = sqrt(MSE);
        MAE = mae / errorPointsCount;
        errorPointsCount++;
        current = ShowError;
    break;
    
    case ShowError:
        Serial.print("Live Performance: "); 
        Serial.println();
        Serial.print("RMSE: ");
        Serial.println(RMSE);
        Serial.print("MSE: ");
        Serial.println(MSE); 
        Serial.print("MAE: ");
        Serial.println(MAE);
        Serial.print("R Squared: ");
        Serial.println(rSquared);
        delay(500);
        current = CollectTestData; 
    break;
    }
    previousBeginTrainingPB = currentBeginTrainingPB;
    previousTestingModePB = currentTestingModePB;
    previousModelValidPB =currentModelValidPB;
    previousTestPB = currentTestPB;
}
