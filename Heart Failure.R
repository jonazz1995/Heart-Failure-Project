library(caret)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(pROC)
library(ROCR)
library(xgboost)
library(rattle)
library(dplyr)
library(MLmetrics)
library(patchwork)
library(lemon)
library(tictoc)
library(randomForest)

file_path <- "~/Desktop/Graduate Life/Machine Learning/Kaggle Datasets"
raw_dataset <- read.csv(paste(file_path,"heart_failure_clinical_records_dataset.csv",sep="/"), header=TRUE) #Load CSV

dataset <- raw_dataset %>%
  mutate(anaemia = if_else(anaemia==1,"YES","NO"),
         diabetes = if_else(diabetes==1,"YES","NO"),
         high_blood_pressure = if_else(high_blood_pressure==1,"YES","NO"),
         sex = if_else(sex==1,"MALE","FEMALE"),
         smoking = if_else(smoking==1,"YES","NO"),
         DEATH_EVENT = if_else(DEATH_EVENT==1,"YES","NO"),
         age = as.integer(age),
         platelets = as.integer(platelets)
  ) %>%
  mutate_if(is.character,as.factor) %>%
  dplyr::select(DEATH_EVENT,anaemia,diabetes,high_blood_pressure,sex,smoking,everything())

set.seed(1995)
training_samples <- createDataPartition(dataset$DEATH_EVENT, p=0.80, list=FALSE)
train_data <- dataset[training_samples,]
test_data <- dataset[-training_samples,]

#Logistic Regression:
set.seed(692)
train_control <- trainControl(method="repeatedcv", number=10)
log_model <- train(DEATH_EVENT~ age + ejection_fraction + serum_creatinine + time, 
                   data=train_data, trControl=train_control, method="glm", family="binomial")
log_pred <- predict(log_model,newdata = test_data)

#K Nearest Neighbors:
set.seed(200)
train_control <- trainControl(method="repeatedcv", number=10)
knn_model <- train(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + time, data = train_data, method = "knn", trControl=train_control, preProcess = c("center", "scale"), tuneLength = 10)
knn_pred <- predict(knn_model, newdata = test_data)

#Support Vector Machine:
set.seed(723)
train_control <- trainControl(method="repeatedcv", number=10, classProbs =  TRUE)
svm_model <- train(DEATH_EVENT ~age + ejection_fraction + serum_creatinine + time, data = train_data, method = "svmLinear", trControl = train_control, preProcess = c("center","scale"))
svm_pred <- predict(svm_model, newdata = test_data)

#SVM using Non-Linear Kernel:
set.seed(139)
train_control <- trainControl(method="repeatedcv", number=10, classProbs =  TRUE)
non_svm_model <- train(DEATH_EVENT~ age + ejection_fraction + serum_creatinine + time, data=train_data, trControl=train_control, method="svmRadial", preProcess=c("center", "scale"), tunelength=10)
non_svm_pred <- predict(non_svm_model, newdata = test_data)

#Decision Tree:
set.seed(823)
train_control <- trainControl(method="repeatedcv", number=10)
tree_model <- train(DEATH_EVENT~age + ejection_fraction + serum_creatinine + time, data=train_data, trControl=train_control, method="rpart")
tree_pred <- predict(tree_model, newdata = test_data)

#Random Forest Ensemble:
set.seed(536)
train_control <- trainControl(method="repeatedcv", number=10)
rf_model <- train(DEATH_EVENT~., data=train_data, trControl=train_control, method="rf")
rf_pred <- predict(rf_model,newdata = test_data)

#Adaboost
set.seed(234)
train_control <- trainControl(method="repeatedcv", number=10)
adabo_model <- train(DEATH_EVENT~age + ejection_fraction + serum_creatinine + time, data=train_data, method="adaboost", trControl=train_control)
adabo_pred <- predict(adabo_model, newdata=test_data)

#Stochastic Gradient Boosting:
set.seed(629)
train_control <- trainControl(method="repeatedcv", number=10, repeats = 10)
gbm_model <- train(DEATH_EVENT~age + ejection_fraction + serum_creatinine + time, data=train_data,
                   trControl=train_control, method="gbm", verbose=0)
gbm_pred <- predict(gbm_model, newdata = test_data)

#Extreme Gradient Boosting:
set.seed(467)
train_control <- trainControl(method="repeatedcv", number=10)
xgb_model <- train(DEATH_EVENT~., data=train_data, trControl=train_control, method="xgbTree")
xgb_pred <- predict(xgb_model,newdata = test_data)

#Neural Networks:
set.seed(888)
train_control <- trainControl(method = 'repeatedcv', number = 10, classProbs = TRUE, verboseIter = FALSE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))
nn_model <- train(DEATH_EVENT~anaemia + age + ejection_fraction + serum_creatinine + time, data = train_data, method = 'nnet', preProcess = c('center', 'scale'), trControl = train_control, metric = "ROC", trace=FALSE)
nn_pred <- predict(nn_model, newdata = test_data)

#Metrics Calculations
pred_list <- list(log_pred,knn_pred,svm_pred,non_svm_pred,tree_pred,rf_pred,adabo_pred,gbm_pred,xgb_pred,nn_pred)
model_list <- list(log_model,knn_model,svm_model,non_svm_model,tree_model,rf_model,adabo_model,gbm_model,xgb_model,nn_model)
algorithms <- c("Logistic Regression","KNN","Linear-SVM","Non-Linear-SVM","Decision Tree","Random Forest",
                "Adaboost","Stochastic Gradient Boosting","XGBoost","Neural Networks")

#Metric Scores
accuracy_scores <- c()
kappa_scores <- c()
recall_scores <- c()
specificity_scores <- c()
precision_scores <- c()
f1_scores <- c()
for(model_pred in pred_list){
  confmat <- confusionMatrix(model_pred, test_data$DEATH_EVENT)
  accuracy_scores <- c(accuracy_scores,round(confmat$overall[["Accuracy"]]*100, digits=2))
  kappa_scores <- c(kappa_scores, round(confmat$overall[['Kappa']]*100, digits = 2))
  recall_scores <- c(recall_scores, round(confmat$byClass[["Sensitivity"]]*100, digits=2))
  specificity_scores <- c(specificity_scores, round(confmat$byClass[["Specificity"]]*100, digits=2))
  precision_scores <- c(precision_scores, round(confmat$byClass[["Precision"]]*100, digits=2))
  f1_scores <- c(f1_scores, round(confmat$byClass[["F1"]]*100, digits=2))
}
# AUC Scores
auc_scores <- c()
for(mod in model_list){
  model_pred <- predict(mod,newdata = test_data, type="prob")
  score <- auc(test_data$DEATH_EVENT,model_pred[,2])
  auc_scores <- c(auc_scores, round(score[[1]]*100, digits=2))
}
metrics_results <- data.frame(algorithms, accuracy_scores, kappa_scores, recall_scores, specificity_scores, precision_scores, f1_scores, auc_scores)
metrics_results$algorithms <- as.factor(metrics_results$algorithms)
names(metrics_results) <- c("Models", "Accuracy", "Kappa", "Recall", "Specificity", "Precision", "F1", "AUC")

library(knitr)
kable(metrics_results)

### Hyper parameter tuning
#KNN

tic()
set.seed(888)
train_control <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, classProbs = TRUE, 
                              verboseIter = FALSE, summaryFunction = twoClassSummary, 
                              preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5), search="grid")
tune_grid <- expand.grid(size=c(3), decay=c(0.1))
nn_model <- train(DEATH_EVENT~anaemia + age + ejection_fraction + serum_creatinine + time, 
                  data = train_data, method = 'nnet', preProcess = c('center', 'scale'), 
                  trControl = train_control, metric = "ROC", trace=FALSE, tuneGrid=tune_grid)
nn_pred <- predict(nn_model, newdata = test_data)
toc()
nn_model$bestTune
confusionMatrix(nn_pred,test_data$DEATH_EVENT)
