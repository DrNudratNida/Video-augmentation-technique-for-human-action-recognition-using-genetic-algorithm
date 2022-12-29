digitDatasetPath ='F:\PhD\experiments\CodeCV\DTMD\CrossValidation\CV_Alexnet';%'F:\PhD\experiments\CodeCV\Muhavi_CNNELM\code\CV_Alexnet\';%'F:\PhD\Firstyear\VideoProj\smallDa
%DataLabel=struct([]);
DataUpdatedGA=struct([]);
%%
% Separate the sets into training and validation data. Use LOOCV 
%Approach for training and test datasets. Randomize it to avoid biasing
%results
%[trainingSets, validationSets] = partition(imgSets, 0.7, 'randomize');

trainingSet = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    
    %%
% Use |countEachLabel| to tabulate the number of images associated with
% each label. In this example, the training set consists of 101 images for
% each of the 10 digits. The test set consists of 12 images per digit.

count=countEachLabel(trainingSet);




%%

    %Display some of the images in the datastore.
trainingImages=trainingSet.Files;
trainingLabels = trainingSet.Labels;
    trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Prepare Training and Validation Image Sets

%minSetCount = min([trainingSet.countEachLabel]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
%trainingSet = partition(imgSets, minSetCount, 'randomize');
trainingSet.countEachLabel


trainingNumFiles = numel(trainingSet.Files);
% rng(1) % For reproducibility
%for i=1:10
[trainActionData,testActionData] = splitEachLabel(trainingSet,0.7,'randomize');
global trainlabel;
global testlabel
 trainlabel = (trainActionData.Labels)';
 testlabel = (testActionData.Labels)';
%DataLabel.trainlabel=trainlabel;
%DataLabel.testlabel=testlabel;
net=alexnet;
layer = 'fc6';
a=numel(trainlabel);
b=numel(testlabel);
trainingFeatures=zeros(a,4096);
testFeatures=zeros(b,4096);
for i=1:a
    img=imread(fullfile(trainActionData.Files{i}));%dirData(i).name
trainingFeatures(i,:) =activations(net,img,layer);
end
for i=1:b
    img=imread(fullfile(testActionData.Files{i}));
testFeatures(i,:) = activations(net,img,layer);
end
trainingFeatures=trainingFeatures';
testFeatures=testFeatures';
% aa=testFeatures(1,:);
% sz = size(aa);
% act1 = cat3,aa,aa,aa);
%Now you can show the activations. Each activation can take any value, so normalize the output using mat2gray. All activations are scaled so that the minimum activation is 0 and the maximum is 1. Display a montage of the 96 images on an 8-by-12 grid, one for each channel in the layer.

%montage(mat2gray(act1),'Size',[8 12])
%%
% Extract the class labels from the training and test data.
traindata=trainingFeatures;%(traindata);%
testdata=testFeatures; %(testdata); %
%traindata=(traindata);%
%testdata=(testdata); %

nn.label = trainlabel;

[traindata,PS] = mapminmax(traindata,-1,1);%
testdata = mapminmax('apply',testdata,PS);

% traindataPca=princomp(traindata');
% traindata=traindataPca*(traindata);
% testdata=traindataPca*testdata;
[trainlabel, testlabel] = label_convertIAVID(single(trainlabel),single(testlabel),'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};
%traindata  =  traindata./( repmat(sqrt(sum(traindata.*traindata)), [size(traindata,1),1]) );
%testdata   =  testdata./(repmat(sqrt(sum(testdata.*testdata)), [size(testdata,1),1]) );

 [traindata,PS] = mapminmax(traindata,-1,1);%
 testdata = mapminmax('apply',testdata,PS);

% [traindata,PS] = mapstd(traindata);%
% testdata = mapstd('apply',testdata,PS);
nn.hiddensize     = 5100;
nn.activefunction = 's';
lamda   = 10e-2;
tol     = 5e-2;
nn.inputsize      = size(traindata,1);
nn.method         = method{1};
nn.type           = 'classification';
%-----------Initialization----------

nn                = elm_initialization(nn);
fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');

nn.method         = method{1};

[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
actualLabels=label_actual_test';
predictedLabels=(label_desired_test)';

order={'interIdle','PtBoardSc','PtStudent','UsingLaptop','UsingPhone','Sitting','Walk','Writing'};

%order={'IdleorInteracting','PtBoardSc','PtStudent','Sitting','UsingLaptop','UsingPhone','Walking','WritingBoard'};
confmat=confusionmat(actualLabels,predictedLabels );
confPlot(confmat,order)
actual=eye(8);
predicted=confmat;
save('crossvalidation_IAVID_Alexnet_0.7.mat','actualLabels','predictedLabels');
% figure
% plotroc(actual,predicted)
%legend(order)
numcorrect = sum(label_actual_test==label_desired_test);
accuracy = (numcorrect/length(testlabel));

accuracy = mean(numcorrect/length(testlabel));
total=length(testlabel);
FitVal=accuracy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  Selection & crossover
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=0;
while (FitVal~=90 && i<=200)%100
   % traindata=Data1.trainingFeatures;%(traindata);%
%testdata=Data1.testFeatures;
matrixA=traindata';
matrixB=testdata';
% size(matrixA);
% size(matrixB);
i=i+1
%matrixA=randi([1 100],[4000 4000]);
pt1=randi([1,2000]);
pt2=randi([2000,4000]);

[poptrain]=MatrixTwoPointCrossOver(matrixA,pt1,pt2);
[poptest]=MatrixTwoPointCrossOver(matrixB,pt1,pt2);
% size(poptrain);
% size(poptest);

[FitVal, traindataUpdated, testdataUpdated]=FitFunc_ELM(poptrain', poptest');
fprintf('      Generation      |    Fitness Value   \n');
fprintf('-------------------------------------------\n');
DataUpdatedGA(i).Generation=i;
DataUpdatedGA(i).FitVal=FitVal;
DataUpdatedGA(i).Trainpop=traindataUpdated;
DataUpdatedGA(i).Testpop=testdataUpdated;
%fprintf(
% Mdl = fitcknn(TwoPtcross,Data1.trainlabel','NumNeighbors',3,'Standardize',1);
% label_desired_r_test=predict(Mdl,TwoPtcrosstest);
% label_actual_r_test=Data1.testlabel';
% 
% confMatrixKNN=confusionmat(label_actual_r_test,label_desired_r_test);
% numcorrect = sum(label_actual_r_test==label_desired_r_test);
% accuracy = numcorrect/length(Data1.testlabel);
% total=length(testlabel);
% 
% [OnePtcross]=MatrixCrossOver(matrixA,pt1);
%i FitVal 
end
GATable=struct2table(DataUpdatedGA);
[maxAcc ind]=max((GATable.FitVal));
%location=GATable(strcmp(cell2mat(GATable.FitVal),maxAcc),:);
r=find((GATable.FitVal)==maxAcc);
location=GATable(ind,:);
OptFeaturesTrain=cell2mat(GATable.Trainpop(ind));
OptFeaturesTest=cell2mat(GATable.Testpop(ind));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              Matrix Crossover
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Mcross]=MatrixCrossOver(matrixA,point)
[r c]=size(matrixA)
Mcross=zeros(r, c);
%point=3;
    for j=1:2:r-1
    [result1 result2] = CrossoverAtPoint(matrixA(j,:), matrixA(j+1,:), point);
    p=j;
    Mcross(p,:)=result1 ;
    Mcross(p+1,:)=result2;
    %j=j+2;
    end
end

function [Mcross]=MatrixTwoPointCrossOver(matrixA,point1,point2)
[r c]=size(matrixA)
Mcross=zeros(r, c);
%point=3;
    for j=1:2:r-1
    [result1 result2] = CrossoverAtTwoPoint(matrixA(j,:), matrixA(j+1,:), point1,point2);
    p=j;
    Mcross(p,:)=result1 ;
    Mcross(p+1,:)=result2;
    %j=j+2;
    end
end

function [result1 result2] = CrossoverAtPoint(sequence1, sequence2, point)
    result1 = [sequence1(1:point-1), sequence2(point:end)];
    result2 = [sequence2(1:point-1), sequence1(point:end)];
end

function [result1 result2] = CrossoverAtTwoPoint(sequence1, sequence2, point1,point2)
%     size(sequence1)
%     size(sequence2)
%     size(sequence1(1:point1-1))
    
    result1 = [sequence1(1:point1-1), sequence2(point1:point2-1),sequence1(point2:end)];
    result2 = [sequence2(1:point1-1), sequence2(point1:point2-1),sequence2(point2:end)];
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Fitness function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [FitVal]=FitFunc_KNNTEST(poptrain, poptest)
global Data1
traindata=poptrain;
testdata=poptest;
%-----------Setting----------------------------------------------------
% traindataPca=princomp(traindata);
% traindata=traindataPca*(traindata);
% testdata=traindataPca*testdata;
 
[traindata,PS] = mapminmax(traindata,-1,1);%
 testdata = mapminmax('apply',testdata,PS);
Mdl = fitcknn(traindata',Data1.trainingLabels','NumNeighbors',3,'Standardize',1);
label_desired_r_test=predict(Mdl,testdata');
label_actual_r_test=Data1.testLabels';

%confMatrixKNN=confusionmat(label_actual_r_test,label_desired_r_test);
numcorrect = sum(label_actual_r_test==label_desired_r_test);
accuracy = numcorrect/length(Data1.testLabels);
FitVal=mean(accuracy)

end

function  [FitVal, traindata, testdata]=FitFunc_ELM(poptrain, poptest)
%global Data1
traindata=poptrain;
testdata=poptest;
%-----------Setting----------------------------------------------------
% traindataPca=princomp(traindata);
% traindata=traindataPca*(traindata);
% testdata=traindataPca*testdata;
 method            = {'ELM','RELM'};

[traindata,PS] = mapminmax(traindata,-1,1);%
 testdata = mapminmax('apply',testdata,PS);
nn.hiddensize     = 2100;
nn.activefunction = 's';
lamda   = 10e-2;
tol     = 5e-2;
nn.inputsize      = size(traindata,1);
nn.method         = method{1};
nn.type           = 'classification';
%-----------Initialization----------

nn                = elm_initialization(nn);
fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');

nn.method         = method{1};
global trainlabel%trainlabel=DataLabel.trainlabel;
global testlabel%=DataLabel.testlabel;

[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
actualLabels=label_actual_test';
predictedLabels=(label_desired_test)';


%confMatrixKNN=confusionmat(label_actual_r_test,label_desired_r_test);
numcorrect = sum(label_actual_test==label_desired_test);
accuracy = (numcorrect/length(testlabel));
FitVal=mean(accuracy)

end


function I = readAndPreprocessImage(filename)
                
        I = imread(filename);
        
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
    end
