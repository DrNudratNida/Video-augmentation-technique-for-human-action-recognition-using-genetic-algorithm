function  [FitVal, traindata, testdata,label_actual_test,label_desired_test]=FitFunc_ELM(poptrain, poptest,trainlabel,testlabel)
%global Data1
traindata=poptrain';
testdata=poptest';
%-----------Setting----------------------------------------------------
% traindataPca=princomp(traindata);
% traindata=traindataPca*(traindata);
% testdata=traindataPca*testdata;
 method            = {'ELM','RELM'};

% [traindata,PS] = mapminmax(traindata,-1,1);%
%  testdata = mapminmax('apply',testdata,PS);
nn.hiddensize     = 129060;
nn.activefunction = 't';
lamda   = 30e-2;
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


%confMatrixKNN=confusionmat(label_actual_r_test,label_desired_r_test);
numcorrect = sum(label_actual_test==label_desired_test);
accuracy = (numcorrect/length(testlabel));
FitVal=mean(accuracy)

end