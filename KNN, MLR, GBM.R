library(data.table)
library(DT)
library(dplyr)
library(class)
library(e1071)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(nnet)



sample_split = function(data,size,n){
  sample_data <- data[sample(1:nrow(data), size*3), ]
  sample_split <-split(sample_data, 
                       rep(1:3, length.out = nrow(sample_data), 
                           each = ceiling(nrow(sample_data)/3)))
  return(sample_split[[n]])
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }


sample_size <- c(500, 1000, 2000)
iterations <- 3



train_data <- read.csv("MNIST-fashion training set-49.csv")
test_data <- read.csv("MNIST-fashion testing set-49.csv")


#MLR
MLR <- function(data) {
  
  mlr.model = multinom(label ~.,
                       data = data)
  
  pred = predict(mlr.model, newdata = test_data)
  return(pred)
  
}



# KNN
KNN <- function(data) {
  knn.model = knn(data[, -1], test_data[, -1], data$label, k = 10)
}


#GBM CV
cvBoosting <- function(data){
  
  trControl = trainControl(method="cv",number=5)
  tuneGrid = expand.grid(n.trees = 125, 
                         interaction.depth = c(1,2,3),
                         shrinkage = (1:100)*0.001,
                         n.minobsinnode=c(5,10,15))
  
  garbage = capture.output(cvModel <- train(label~.,
                                            data=data,
                                            method="gbm",
                                            trControl=trControl, 
                                            tuneGrid=tuneGrid))
  cvBoost = gbm(label ~., 
                data = data,
                distribution = "multinomial",
                n.trees=cvModel$bestTune$n.trees,
                interaction.depth=cvModel$bestTune$interaction.depth,
                shrinkage=cvModel$bestTune$shrinkage,
                n.minobsinnode = cvModel$bestTune$n.minobsinnode)
  
  pred = predict(cvBoost, newdata=test, n.trees = 125)
  return(pred)
}




models = c( MLR, KNN)


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
      C = (1- sum(test_data$label == pred)/NROW(test_data$label))
      Points = 0.15 * A + 0.1 * B + 0.75 * C
      score_row = data.frame(Model, 'Sample Size' = sample_size[i], Data, A, B, C, Points)
      scoreboard = rbind(scoreboard,score_row)
    }
  }}
print(scoreboard)
