library(dplyr)
library(class)
library(e1071)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

set.seed(123)

sample_split = function(data,size,n){
  sample_data <- data[sample(1:nrow(data), size*3), ]
  sample_split <-split(sample_data, 
                       rep(1:3, length.out = nrow(sample_data), 
                           each = ceiling(nrow(sample_data)/3)))
  return(sample_split[[n]])
  }

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

train_data = read.csv('train_data.csv')
test_data = read.csv('test_data.csv')
sample_size = c(5000, 10000, 20000)

dat_500_1 = sample_split(train_data, sample_size[1], 1)


SVM = function(data){
  model = svm(as.factor(label) ~ ., data = data, kernel = "linear", scale = T)
  pred = predict(model, test_data[,-1])
  return(pred)
}

XGB = function(data){
  train.label = as.integer(as.factor(data$label))-1
  train_matrix = as.matrix(data[,-1])
  test.label = as.integer(as.factor(test_data$label))-1
  test_matrix = as.matrix(test_data[,-1])
  xgb.train = xgb.DMatrix(data=train_matrix,label=train.label)
  xgb.test = xgb.DMatrix(data=test_matrix,label=test.label)
  model = xgboost(data = xgb.train, max.depth = 30, eta = 0.001, 
                  nthread = 2, nrounds = 2, num_class = length(unique(data$label)), objective = "multi:softprob")
  pred <- predict(model, newdata = xgb.test,reshape=T)
  pred = as.data.frame(pred)
  colnames(pred) = levels(as.factor(test_data$label))
  pred$prediction = apply(pred,1,function(x) colnames(pred)[which.max(x)])
  pred$label = levels(as.factor(test_data$label))[train.label+1]
  return(pred$prediction)
}



models = c(SVM, XGB) ## add your models

scoreboard = data.frame()
for (k in 1:length(models)) {
  for (i in 1:length(sample_size)) {
    for (j in 1:3) {
      train = sample_split(train_data, sample_size[i], j)
      start_time <- Sys.time()
      pred = models[[k]](train)
      end_time <- Sys.time()
      sys_time = end_time - start_time
      Model = paste('Model',k)
      Data = paste('dat_',sample_size[i],'_',j, sep = '')
      A = sample_size[i]/60000
      B = min(1,sys_time/60)
      C = sum(test_data$label == pred)/NROW(test_data$label)
      Points = 0.15 * A + 0.1 * B + 0.75 * C
      score_row = data.frame(Model, 'Sample Size' = sample_size[i], Data, A, B, C, Points)
      scoreboard = rbind(scoreboard,score_row)
  }
}}
print(scoreboard)

