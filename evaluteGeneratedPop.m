function [accuracy UpdatedF]=evaluteGeneratedPop(stateChromosome,mdl)
SVMModel=mdl;
UpdatedF=stateChromosome(:,2);
UpdatedF=table2cell(UpdatedF);
UpdatedF=cell2mat(UpdatedF);
label_desired_r_test=predict(SVMModel,UpdatedF);
label=table2array(stateChromosome(:,3));
label_actual_r_test=label;
numcorrect = sum(label_actual_r_test==label_desired_r_test);
accuracy = sum(numcorrect/length(label));
end


