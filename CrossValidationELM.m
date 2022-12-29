digitDatasetPath ='F:\PhD\experiments\CodeCV\DTMD\CrossValidation\CV_Alexnet\';

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

figure;
%perm = randperm(100,20);
for i = 1:20
    subplot(4,5,i);
    imshow(trainingSet.Files{i});
end
%CountLabel = trainingSet.countEachLabel;
img = readimage(trainingSet,1);
size(img)
trainingNumFiles = numel(trainingSet.Files);
rng(1) % For reproducibility
%for i=1:10 

[trainActionData,testActionData] = splitEachLabel(trainingSet,0.7,'randomize');
net=alexnet;
%layer = 'fc8';
%a=numel(trainIndex);
%b=numel(testIndex);
%trainingFeatures=zeros(a,1000);
%testFeatures=zeros(b,1000);

trainlabel = (trainActionData.Labels)';
testlabel = (testActionData.Labels)';
tic
net=alexnet;
layer = 'fc6';
a=numel(trainlabel);
b=numel(testlabel);
trainingFeatures=zeros(a,4096);
testFeatures=zeros(b,4096);

%testFeatures=testFeatures';
% aa=testFeatures(1,:);
% sz = size(aa);
% act1 = cat3,aa,aa,aa);
%Now you can show the activations. Each activation can take any value, so normalize the output using mat2gray. All activations are scaled so that the minimum activation is 0 and the maximum is 1. Display a montage of the 96 images on an 8-by-12 grid, one for each channel in the layer.

%montage(mat2gray(act1),'Size',[8 12])
%%
% Extract the class labels from the training and test data.
traindata=trainingFeatures;%(traindata);%
testdata=testFeatures; %(testdata); %

tic
nn.label = trainlabel;
for i=1:a
    img=imread(fullfile(trainActionData.Files{i}));%dirData(i).name
trainingFeatures(i,:) =activations(net,img,layer);
end
% for i=1:b
%     img=imread(fullfile(testActionData.Files{i}));
% testFeatures(i,:) = activations(net,img,layer);
% end

[trainlabel, testlabel] = label_convertIAVID(single(trainlabel),single(testlabel),'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};
nn.hiddensize     = 1300;
nn.activefunction = 's';
lamda   = 10e-2;
tol     = 5e-2;
nn.inputsize      = size(traindata,1);
nn.method         = method{1};
nn.type           = 'classification';
%-----------Initialization----------
trainingdata=trainingFeatures';

nn                = elm_initialization(nn);
fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');

nn.method         = method{1};

[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
trainTime=toc;
tic
for i=1:b
    img=imread(fullfile(testActionData.Files{i}));
testFeatures(i,:) = activations(net,img,layer);
end
testFeatures=testFeatures';
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
actualLabels=label_actual_test';
predictedLabels=(label_desired_test)';
testTime=toc;
order={'interIdle','PtBoardSc','PtStudent','UsingLaptop','UsingPhone','Sitting','Walk','Writing'};

%order={'IdleorInteracting','PtBoardSc','PtStudent','Sitting','UsingLaptop','UsingPhone','Walking','WritingBoard'};
confmat=confusionmat(actualLabels,predictedLabels );
confPlot(confmat,order)
actual=eye(8);
predicted=confmat;
%save('crossvalidation_IAVID_Alexnet_0.7.mat','actualLabels','predictedLabels');
figure
%plotroc(actual,predicted)
%legend(order)
[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
%predicted=predict(categoryClassifier, validationSets);
actualLabels=label_actual_test';
predictedLabels=(label_desired_test)';
%confMatrix2= evaluate(categoryClassifier, validationSets);
%colormap(confMatrix);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Saving features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
Train_Features=traindata;
Test_Features=testdata;
Train_Label=trainlabel;
Test_Label=testlabel;
%save('LOAO_Alexnet_fc8_IAVID_1300.mat','predictedLabels','actualLabels','j','cM');
numcorrect = sum(label_actual_test==label_desired_test);
accuracy = (numcorrect/length(testlabel));

accuracy = mean(numcorrect/length(testlabel));
total=length(testlabel);
FitVal=accuracy;


% NumGeneration=100;
% tic
% [GATable, OptFeaturesTrain,OptFeaturesTest,location,maxFitness]=EvolutionaryEDTMD(NumGeneration,FitVal,traindata,testdata,trainlabel,testlabel);%figure
% [nn, GAacc_train,GAlabel_actual,GAlabel_desired]   = elm_train(OptFeaturesTrain, trainlabel, nn);
% toc
% [nn1, GAacc_test,GAlabel_actual_test,GAlabel_desired_test]    = elm_test(OptFeaturesTest, testlabel, nn);
% GAactual=GAlabel_actual_test';
% GApredicted=(GAlabel_desired_test)';
% cM=confmat;
% %save(g,'GATable','OptFeaturesTrain','OptFeaturesTest','location','maxFitness','GAactualLabels','GApredictedLabels');
% for i =1:size(cM,1)
%  
%      precision(i)=cM(i,i)/sum(cM(:,i));
%  end
%  for i =1:size(cM,1)
% % 
%      Recall(i)=cM(i,i)/sum(cM(i,:));
%  end
% % 
%  Precision=sum(precision)/size(cM,1);
%  Recall=mean(Recall);
% % 
% % %%% F-score
% % f=2*1*0.967/1.967;
%  F_score=2*Recall*Precision/(Precision+Recall);
%  %F_score=2*1/((1/Precision)+(1/Recall));
%  error=1-F_score;
% numcorrectN = sum((actual)==(predicted));
% accuracyN = mean(numcorrectN/length(actual));
% %accuracyN = mean(numcorrect/length(actual));
% %plotroc((actual),(predicted));
% %ACC=mean(diag(cM))/length(testlabel);
% %numcorrect = sum(actual==predicted);
% %accuracy = numcorrect/length(testlabel);
% %[precision recall f1Scores  meanF1]= Evaluation(cM)
% % GApredicted=[GApredictedLabels{1,1}',GApredictedLabels{2,1}',GApredictedLabels{3,1}',GApredictedLabels{4,1}',GApredictedLabels{5,1}',GApredictedLabels{6,1}',GApredictedLabels{7,1}',GApredictedLabels{8,1}',GApredictedLabels{9,1}',GApredictedLabels{10,1}',GApredictedLabels{11,1}',GApredictedLabels{12,1}'];
% % %order={'ClimbLadder','CrawlOnKnees','DrawGraffiti','DrunkWalk','JumpOverFence','JumpOverGap','Kick','LookInCar','PickupThrowObject','PullHeavyObject','Punch','RunStop','ShotGunCollapse','SmashObject','WalkFall','WalkTurnBack','WaveArms'};
% % %predicted=order(predicted);
% % GAactual=[GAactualLabels{1,1}',GAactualLabels{2,1}',GAactualLabels{3,1}',GAactualLabels{4,1}',GAactualLabels{5,1}',GAactualLabels{6,1}',GAactualLabels{7,1}',GAactualLabels{8,1}',GAactualLabels{9,1}',GAactualLabels{10,1}',GAactualLabels{11,1}',GAactualLabels{12,1}'];
% %actual=order(actual);
% GAcM=confusionmat((GAactual),(GApredicted));
% figure
% confPlot(GAcM,order);
% numcorrectGA = sum((GAactual)==(GApredicted));
% accuracyGA = (numcorrectGA/length(GAactual));
% 
% 
%  plotroc(GAactual,GApredicted)
% % s=confusionmat(a,b)
% for i =1:size(GAcM,1)
%  
%      GAprecision(i)=GAcM(i,i)/sum(GAcM(:,i));
%  end
%  for i =1:size(GAcM,1)
% % 
%      GARecall(i)=GAcM(i,i)/sum(GAcM(i,:));
%  end
%  GARecall=mean(GARecall);
% GAPrecision=sum(GAprecision)/size(GAcM,1);
% % 
% % %%% F-score
% % f=2*1*0.967/1.967;
%  GAF_score=2*GARecall*GAPrecision/(GAPrecision+GARecall);
%  %GAF_score=2*1/((1/GAPrecision)+(1/GARecall));
%  GAerror=1-GAF_score;


function I = readAndPreprocessImage(filename)
                
        I = imread(filename);
        
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
    end
