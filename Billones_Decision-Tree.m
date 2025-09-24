data = readtable('Social_Network_Ads(8).csv');
    % Standardization
    stand_age = (data.Age - mean(data.Age)) / std(data.Age);
    data.Age = stand_age;
    stand_estimated_salary = (data.EstimatedSalary - mean(data.EstimatedSalary)) / std(data.EstimatedSalary);
    data.EstimatedSalary = stand_estimated_salary;

    %classification_model = fitcknn(data, 'Purchased~Age+EstimatedSalary');
    %classification_model = fitcnb(data, 'Purchased~Age+EstimatedSalary', 'Distribution', 'kernel');
    % classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary')

    % classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary', MinParentSize=20);
    % classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary', MinLeafSize=20);
    % classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary', SplitCriterion='gdi');
    % classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary', SplitCriterion='twoing');
    classification_model = fitctree(data, 'Purchased~Age+EstimatedSalary', SplitCriterion='deviance');


    cv = cvpartition(classification_model.NumObservations,'HoldOut',0.3);

    cross_validated_model = crossval(classification_model,'cvpartition',cv);

    Predictions = predict(cross_validated_model.Trained{1}, data(test(cv),1:end-1));

    Results = confusionmat(cross_validated_model.Y(test(cv)), Predictions);

    %% -------------- Visualizing training set results --------------
% ---------------------------- Code ---------------------------
 
labels = unique(data.Purchased);
classifier_name = 'K-Nearest Neigbor (Training Results)';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
training_data = data(training(cv),:);
Y = ismember(training_data.Purchased,labels{1});
 
 
scatter(training_data.Age(Y),training_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~Y),training_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
 
xlabel('Age');
ylabel('Estimated Salary');
 
title(classifier_name);
legend off, axis tight
 
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
%% -------------- Visualizing testing set results ----------------
% ---------------------------- Code ---------------------------
 
labels = unique(data.Purchased);
classifier_name = 'K-Nearest Neigbor (Testing Results)';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
figure
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
testing_data = data(test(cv),:);
Y = ismember(testing_data.Purchased,labels{1});
 
scatter(testing_data.Age(Y),testing_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Age(~Y),testing_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Age');
ylabel('Estimated Salary');
title(classifier_name);
legend off, axis tight
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
 
%________________________________________________________________

disp('Confusion Matrix:');
disp(Results);

TP = Results(2,2);

TN = Results(1,1);

FP = Results(1,2);

FN = Results(2,1);

accu = (TP + TN) / (TP +TN +FP +FN);
preci = TP / (TP + FP);
rcall = TP / (TP + FN);
f1_score = 2 * (preci * rcall) / (preci + rcall);

fprintf('Accuracy = %.2f%%', accu * 100);
fprintf('\nPrecision = %.2f%%', preci * 100);
fprintf('\nRecall = %.2f%%', rcall * 100);
fprintf('\nF1 Score = %.2f%%', f1_score * 100);
fprintf('\n');
