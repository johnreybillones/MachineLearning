clear all
%% --------------- Importing the dataset -------------------------
% ---------------------------- Code ---------------------------
data = readtable('Social_Network_Ads(8).csv');
%% -------------- Feature Scalling -------------------------------
% -------------- Method 1: Standardization ----------------------
% ---------------------------- Code -----------------------------
stand_age = (data.Age - mean(data.Age))/std(data.Age);
data.Age = stand_age;
stand_estimted_salary = (data.EstimatedSalary - mean(data.EstimatedSalary))/std(data.EstimatedSalary);
data.EstimatedSalary = stand_estimted_salary;
%%%%---------------Classifying Data -----------------------------
%% -------------- Building Classifier ----------------------------
classification_model = fitcsvm(data,'Purchased~Age+EstimatedSalary');
%Kernel Function to test
%classification_model_1 = fitcsvm(data,'Purchased~Age+EstimatedSalary','KernelFunction','linear');
classification_model_1 = fitcsvm(data,'Purchased~Age+EstimatedSalary','KernelFunction','rbf');
%classification_model_1 = fitcsvm(data,'Purchased~Age+EstimatedSalary','KernelFunction','polynomial');
% Test for Linear, gaussian and OutlierFraction, 0.1

%% -------------- Test and Train sets ----------------------------
% ---------------------------- Code ---------------------------
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);
cross_validated_model = crossval(classification_model,'cvpartition',cv);
cross_validated_model_1 = crossval(classification_model_1,'cvpartition',cv);
%% -------------- Making Predictions for Test sets ---------------
% ---------------------------- Code ---------------------------
Predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1));
Predictions_1 = predict(cross_validated_model_1.Trained{1},data(test(cv),1:end-1));
%% -------------- Analyzing the predictions ---------------------
% ---------------------------- Code ---------------------------
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions);
Results_1 = confusionmat(cross_validated_model_1.Y(test(cv)),Predictions_1);
% %% -------------- Visualizing training set results --------------
labels = unique(data.Purchased);
classifier_name = 'SVM (Default options)';
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
% %% -------------- Visualizing training set results --------------
% % ---------------------------- Code ---------------------------
labels = unique(data.Purchased);
classifier_name = 'SVM (Modified options)';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(cross_validated_model_1.Trained{1},XGrid);
figure
xxx = ismember(predictions_meshgrid,'Not Purchased');
scatter(XGrid(xxx,1),XGrid(xxx,2), 'o' , 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');
hold on
scatter(XGrid(~xxx,1),XGrid(~xxx,2), 'o' , 'MarkerEdgeColor', 'green', 'MarkerFaceColor', 'green');
% gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
%
% hold on
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
classifier_name = 'K-Nearest Neigbor';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
test_data = data(test(cv),:);
Y = ismember(test_data.Purchased,labels{1});
scatter(test_data.Age(Y),test_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(test_data.Age(~Y),test_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
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
