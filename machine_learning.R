# Machine learning model attempt with Cai's data
setwd("D:/Cai_paper/")
library(xlsx)
library(dplyr)
library(ggvis)

dat_24 <- read.csv('data_raw.csv')
dat_lt <- read.xlsx('LT_TMX.xlsx', sheetIndex = 1)



colnames(dat_lt)[1] <- 'Group'
colnames(dat_lt)[9]<- 'ERA'
colnames(dat_lt)[10]<- 'ERB'

colnames(dat_24)[c(1,14)] <- c('Group', 'ERA')


dat_lt <- dat_lt[,c(1,2,5:10)]
dat_24 <- dat_24[,c(1,4,9:12, 14,15)]


subst.mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

dat_lt_mean <- dat_lt %>% 
  group_by(Group) %>% 
  mutate(Exploration.Proportion = subst.mean(Exploration.Proportion),
         Cresyl.Violet = subst.mean(Cresyl.Violet),
         Caspase.3 = subst.mean(Caspase.3),
         GFAP = subst.mean(GFAP),
         Iba1 = subst.mean(Iba1),
         ERA = subst.mean(ERA),
         ERB = subst.mean(ERB))

dat_24_mean <- dat_24 %>% 
  group_by(Group) %>% 
  mutate(Exploration.Proportion = subst.mean(Exploration.Proportion),
         Cresyl.Violet = subst.mean(Cresyl.Violet),
         Caspase.3 = subst.mean(Caspase.3),
         GFAP = subst.mean(GFAP),
         Iba1 = subst.mean(Iba1),
         ERA = subst.mean(ERA),
         ERB = subst.mean(ERB))

dat_24_mean <- dat_24_mean %>% 
  ungroup() %>% 
  mutate(Group = ifelse(as.character(Group) == 'Sham+Veh', 'Sh-Veh',
                        ifelse(as.character(Group) == 'Sham+TMX', 'Sh-TMX', 
                               ifelse(as.character(Group) == 'SI+Veh', 'SI-Veh', 'SI-TMX'))))




dat_24_mean$group <- NA
dat_24_mean$group[c(which(dat_24_mean$Group == 'Sh-Veh'), which(dat_24_mean$Group == 'Sh-TMX'), which(dat_24_mean$Group == 'SI-TMX'))] <- 0
dat_24_mean$group[which(dat_24_mean$Group == 'SI-Veh')] <- 1

dat_lt_mean$group <- NA
dat_lt_mean$group[c(which(dat_lt_mean$Group == 'Sh-Veh'), which(dat_lt_mean$Group == 'Sh-TMX'), which(dat_lt_mean$Group == 'SI-TMX'))] <- 0
dat_lt_mean$group[which(dat_lt_mean$Group == 'SI-Veh')] <- 1


nor <-function(x) { (x -min(x))/(max(x)-min(x))   }


dat_total <- rbind(as.data.frame(dat_24_mean), as.data.frame(dat_lt_mean))
dat_total_norm <- as.data.frame(lapply(dat_total[,c(2:8)], nor))


##extract training set
train <- dat_total_norm[1:43,]
train$group <- dat_total[1:43,9]
rows <- sample(nrow(train))
train <- train[rows,]
##extract testing set
test <- dat_total_norm[44:90,] 
test$group <- dat_total[44:90,9]
rows <- sample(nrow(test))
test <- test[rows,]
##extract 5th column of train dataset because it will be used as 'cl' argument in knn function.
target_category <- dat_total[1:43,9]
##extract 5th column if test dataset to measure the accuracy
test_category <- dat_total[44:90,9]

library(class)
##run knn function
pr <- knn(data_balanced_both,data.test.both,cl=data_balanced_both[,8],k=2)

##create confusion matrix
tab <- table(pr,data.test.both[,8])

##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)



roc.curve(data.test.both$group, pr)
accuracy.meas(data.test.both$group, pr)







vec <- vector(mode = 'numeric', length = 20)

for(i in 1:20){
  #pr <- knn(train[,c(2,6)],test[,c(2,6)],cl=target_category,k=i)
  pr <- knn(data_balanced_over[,-7],data.test[,-7], cl=data_balanced_over[,7],k=i)
  tab <- table(pr,data.test[,7])
  vec[i] <- accuracy(tab)
}

plot(vec, type="b", xlab="K- Value",ylab="Accuracy level")

pr <- knn(train[,c(2,6)],test[,c(2,6)],cl=target_category,k=1)
tab <- table(pr,test_category)
accuracy(tab)


data_balanced_over %>% ggvis(~Caspase.3, ~GFAP, fill = ~group) %>% layer_points()
train %>% ggvis(~Caspase.3, ~ERA, fill = ~target_category) %>% layer_points()
test %>% ggvis(~Caspase.3, ~ERB, fill = ~test_category) %>% layer_points()

ggplot(train, (aes(ERA))) + geom_point(aes(color = group), size = 4, alpha = 0.4)
ggplot(train, aes(ERA)) + geom_histogram(aes(fill = as.factor(group)), color = 'black', bins = 15, alpha = 0.5) + theme_bw()
ggplot(test, aes(ERA)) + geom_histogram(aes(fill = as.factor(group)), color = 'black', bins = 15, alpha = 0.4) + theme_bw()
predicted.group <- NULL
error.rate <- NULL

for(i in 1:10){
  predicted.group <- knn(data_balanced_over[,-7],data.test[,-7], cl=data_balanced_over[,7],k=i)
  error.rate[i] <- mean(data.test[,7] != predicted.group)
}

library(ggplot2)

k.values <- 1:10
error.df <- data.frame(error.rate, k.values)
ggplot(error.df, aes(x = k.values, y=error.rate)) + geom_point() + geom_line(lty = 'dotted', color = 'blue')



library(randomForest)
# Perform training:
rf_classifier = randomForest(as.factor(target_category) ~ ., data=train, ntree=100, mtry=2, importance=TRUE)


rf_classifier


# Dealing with classes imbalance
library(ROSE)
library(rpart)
library(rpart.plot)

data_balanced_over <- ovun.sample(group ~ ., data = train, method = "over",N = 64)$data
data.test.over <- ovun.sample(group ~ ., data = test, method = "over",N = 70)$data

table(data_balanced_over$group)
table(data.test.over$group)
table(test$group)
table(train$group)


data_balanced_under <- ovun.sample(group ~ ., data = train, method = "under", N = 22, seed = 1)$data
data.test.under <- ovun.sample(group ~ ., data = test, method = "under",N = 24, seed = 1)$data
table(data_balanced_under$group)
table(data.test.under$group)
table(test$group)
table(train$group)









tree.over <- rpart(group ~ ., data = data_balanced_over)
pred.tree.over <- predict(tree.over, newdata = data.test.over)

roc.curve(data.test.over$group, pred.tree.over)
accuracy.meas(data.test.over$group, pred.tree.over)



tree.over <- rpart(group ~ ., data = data_balanced_under)
pred.tree.over <- predict(tree.over, newdata = data.test.under)

roc.curve(data.test.under$group, pred.tree.over)
accuracy.meas(data.test.under$group, pred.tree.over)






data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43, seed = 1)$data
rows <- sample(nrow(data_balanced_both))
data_balanced_both <- data_balanced_both[rows,]


data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, seed = 1, N = 47)$data
rows <- sample(nrow(data.test.both))
data.test.both <- data.test.both[rows,]



for_csv <- rbind(data_balanced_both,data.test.both)
for_csv %>% 
  mutate(group)


write.csv(,"staining.csv")

table(data_balanced_both$group)
table(data.test.both$group)
table(test$group)
table(train$group)


tree.both <- rpart(group ~ ., method = 'class', data = data_balanced_both[,c(5,7)])
pred.tree.both <- predict(tree.both, newdata = data.test.both[,c(5,7)])
prp(tree.both)


joiner <- function(x){
  if(x > 0.5){
    return(1)
  } else {
    return(0)
  }
}





pred.tree.both <- as.data.frame(pred.tree.both)
pred.tree.both$con <- sapply(pred.tree.both$`1`, joiner)


roc.curve(data.test.both$group, pred.tree.both$con)
accuracy.meas(data.test.both$group, pred.tree.both$con)






predicted.group <- NULL
error.rate <- NULL

for(i in 1:10){
  predicted.group <- knn(data_balanced_both[,-7],data.test.both[,-7], cl=data_balanced_both[,7],k=i)
  error.rate[i] <- mean(data.test.both[,7] != predicted.group)
}


k.values <- 1:10
error.df <- data.frame(error.rate, k.values)
ggplot(error.df, aes(x = k.values, y=error.rate)) + geom_point() + geom_line(lty = 'dotted', color = 'blue')


library(randomForest)
rf_classifier = randomForest(as.factor(group) ~ ., data=data_balanced_both[,c(1,3,6,8)], ntree=300, mtry=1,importance=TRUE)
rf_classifier$confusion
summary(rf_classifier)

rf.preds <- predict(rf_classifier, data.test.both[,c(1,3,6)])

rf.preds <- as.data.frame(rf.preds)



roc.curve(data.test.both$group, rf.preds$rf.preds)
accuracy.meas(data.test.both$group, rf.preds$rf.preds)

table(rf.preds$rf.preds, data.test.both[,8])


library(caret)

data_balanced_both$group <- as.factor(data_balanced_both$group)
data.test.both$group <- as.factor(data.test.both$group)


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(group ~., data = data_balanced_both[,c(5,7)], method = "svmLinear",
                    trControl=trctrl,
                    tuneLength = 10)
svm_Linear


test_pred <- predict(svm_Linear, newdata = data.test.both[,c(5,7)])
a <- confusionMatrix(table(test_pred, data.test.both[,7]))



library(gtools)

x <- c(1:7)
data_balanced_both$group <- as.factor(data_balanced_both$group)

matr.comb <- combinations(n=7,r=1,v=x)
matr.comb <- cbind(matr.comb, rep(8, nrow(matr.comb)))  


List.res1 <- vector("list", nrow(matr.comb))


for(i in 1:nrow(matr.comb)){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[i,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[i,]])
  a <- confusionMatrix(table(test_pred, data.test.both[,8]))
  List.res1[[i]] <- a
  names(List.res1)[i] <- paste(matr.comb[i,], collapse = " - ")
}


mertics1 <- data.frame(x = names(sapply(List.res1, function(x)  x$overall[2])),
                      kappa = sapply(List.res1, function(x)  x$overall[2]),
                      accuracy = sapply(List.res1, function(x)  x$overall[1]),
                      specif = sapply(List.res1, function(x)  x$byClass[2]),
                      sens = sapply(List.res1, function(x)  x$byClass[1]),
                      recall = sapply(List.res1, function(x)  x$byClass[6]),
                      f1 = sapply(List.res1, function(x)  x$byClass[7]))

ggplot(mertics1, aes(x = as.character(x))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 1")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))




matr.comb <- combinations(n=6,r=2,v=x)
matr.comb <- cbind(matr.comb, rep(8, nrow(matr.comb)))  


List.res <- vector("list", nrow(matr.comb))


for(i in 1:nrow(matr.comb)){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[i,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[i,]])
  a <- confusionMatrix(table(test_pred, data.test.both[,8]))
  List.res[[i]] <- a
  names(List.res)[i] <- paste(matr.comb[i,], collapse = " - ")
}


mertics <- data.frame(x = names(sapply(List.res, function(x)  x$overall[2])),
                     kappa = sapply(List.res, function(x)  x$overall[2]),
                     accuracy = sapply(List.res, function(x)  x$overall[1]),
                     specif = sapply(List.res, function(x)  x$byClass[2]),
                     sens = sapply(List.res, function(x)  x$byClass[1]),
                     recall = sapply(List.res, function(x)  x$byClass[6]),
                     f1 = sapply(List.res, function(x)  x$byClass[7]))

ggplot(mertics, aes(x = as.character(x))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 2")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))





matr.comb <- combinations(n=6,r=3,v=x)
matr.comb <- cbind(matr.comb, rep(8, nrow(matr.comb)))  


List.res.3 <- vector("list", nrow(matr.comb))


for(i in 1:nrow(matr.comb)){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[i,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[i,]])
  a <- confusionMatrix(table(test_pred, data.test.both[,8]))
  List.res.3[[i]] <- a
  names(List.res.3)[i] <- paste(matr.comb[i,], collapse = " - ")
}


mertics.3 <- data.frame(x = names(sapply(List.res.3, function(x)  x$overall[2])),
                      kappa = sapply(List.res.3, function(x)  x$overall[2]),
                      accuracy = sapply(List.res.3, function(x)  x$overall[1]),
                      specif = sapply(List.res.3, function(x)  x$byClass[2]),
                      sens = sapply(List.res.3, function(x)  x$byClass[1]),
                      recall = sapply(List.res.3, function(x)  x$byClass[6]),
                      f1 = sapply(List.res.3, function(x)  x$byClass[7])
)


ggplot(mertics.3, aes(x = as.character(x))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 3")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))




matr.comb <- combinations(n=6,r=4,v=x)
matr.comb <- cbind(matr.comb, rep(8, nrow(matr.comb)))  


List.res.4 <- vector("list", nrow(matr.comb))


for(i in 1:nrow(matr.comb)){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[i,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[i,]])
  a <- confusionMatrix(table(test_pred, data.test.both[,8]))
  List.res.4[[i]] <- a
  names(List.res.4)[i] <- paste(matr.comb[i,], collapse = " - ")
}


mertics.4 <- data.frame(x = names(sapply(List.res.4, function(x)  x$overall[2])),
                        kappa = sapply(List.res.4, function(x)  x$overall[2]),
                        accuracy = sapply(List.res.4, function(x)  x$overall[1]),
                        specif = sapply(List.res.4, function(x)  x$byClass[2]),
                        sens = sapply(List.res.4, function(x)  x$byClass[1]),
                        recall = sapply(List.res.4, function(x)  x$byClass[6]),
                        f1 = sapply(List.res.4, function(x)  x$byClass[7])
)


ggplot(mertics.4, aes(x = as.character(x))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 4")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))




matr.comb <- combinations(n=6,r=5,v=x)
matr.comb <- cbind(matr.comb, rep(7, nrow(matr.comb)))  
List.res.5 <- vector("list", nrow(matr.comb))


for(i in 1:nrow(matr.comb)){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[i,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[i,]])
  a <- confusionMatrix(table(test_pred, data.test.both[,7]))
  List.res.5[[i]] <- a
  names(List.res.5)[i] <- paste(matr.comb[i,], collapse = " - ")
}


mertics.5 <- data.frame(x = names(sapply(List.res.5, function(x)  x$overall[2])),
                        kappa = sapply(List.res.5, function(x)  x$overall[2]),
                        accuracy = sapply(List.res.5, function(x)  x$overall[1]),
                        specif = sapply(List.res.5, function(x)  x$byClass[2]),
                        sens = sapply(List.res.5, function(x)  x$byClass[1]),
                        recall = sapply(List.res.5, function(x)  x$byClass[6]),
                        f1 = sapply(List.res.5, function(x)  x$byClass[7])
)


ggplot(mertics.5, aes(x = as.character(x))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 5")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))



# Specificity
# True Negative Rate is defined as TN / (FP+TN). 
# False Positive Rate corresponds to the proportion of negative data points that are correctly considered as negative, 
# with respect to all negative data points.


# True Positive Rate (Sensitivity) : True Positive Rate is defined as TP/ (FN+TP). 
# True Positive Rate corresponds to the proportion of positive data points that are correctly considered as positive, 
# with respect to all positive data points.



# Recall : It is the number of correct positive results divided by the number of all relevant samples 
# (all samples that should have been identified as positive).



matr.comb <- combinations(n=6,r=4,v=x)
matr.comb <- cbind(matr.comb, rep(7, nrow(matr.comb)))  



kappas <- vector('numeric', length = 30)
acc <- vector('numeric', length = 30)
spec <- vector('numeric', length = 30)
sens <- vector('numeric', length = 30)
rec <- vector('numeric', length = 30)
f1 <- vector('numeric', length = 30)

for(i in 1:30){
  svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[12,]], method = "svmLinear",
                      trControl=trctrl,
                      tuneLength = 10)
  test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[12,]])
  kappas[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$overall[2]
  acc[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$overall[1]
  spec[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$byClass[2]
  sens[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$byClass[1]
  rec[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$byClass[6]
  f1[i] <- confusionMatrix(table(test_pred, data.test.both[,7]))$byClass[7]
}

mean(kappas)







library(e1071)
?svm
svm.fit <- svm(group ~., data = data_balanced_both[,c(5,7)], kernel = 'linear', cost = 10, scale=F)
plot(svm.fit,data_balanced_both[,c(5,7)] )
plot(model.frame(svm.fit)[,2], col = predict(svm.fit))














f2 <- data.frame(  kappa = rep(0,15),
                             accuracy = rep(0,15),
                             specif = rep(0,15),
                             sens = rep(0,15),
                             recall = rep(0,15),
                             f1 = rep(0,15))
f1 <- data.frame(  kappa = rep(0,7),
                   accuracy = rep(0,7),
                   specif = rep(0,7),
                   sens = rep(0,7),
                   recall = rep(0,7),
                   f1 = rep(0,7))

f3 <- data.frame(  kappa = rep(0,20),
                   accuracy = rep(0,20),
                   specif = rep(0,20),
                   sens = rep(0,20),
                   recall = rep(0,20),
                   f1 = rep(0,20))

f4 <- data.frame(  kappa = rep(0,15),
                   accuracy = rep(0,15),
                   specif = rep(0,15),
                   sens = rep(0,15),
                   recall = rep(0,15),
                   f1 = rep(0,15))

list.results <- rep(list(list(f1,f2,f3,f4)),30)

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
x <- 1:7
for(i in 1:30){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43, seed = 1)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, seed = 1, N = 47)$data
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  for(j in 1:4){
    matr.comb <- combinations(n=7,r=j,v=x)
    matr.comb <- cbind(matr.comb, rep(8, nrow(matr.comb)))  
    List.res <- vector("list", nrow(matr.comb))
    for(k in 1:nrow(matr.comb)){
      svm_Linear <- train(group ~., data = data_balanced_both[,matr.comb[k,]], method = "svmLinear",
                          trControl=trctrl,
                          tuneLength = 10)
      test_pred <- predict(svm_Linear, newdata = data.test.both[,matr.comb[k,]])
      a <- confusionMatrix(table(test_pred, data.test.both[,8]))
      List.res[[k]] <- a
      names(List.res)[k] <- paste(matr.comb[k,], collapse = " - ")
    }
    mertics <- data.frame(  kappa = sapply(List.res, function(x)  x$overall[2]),
                            accuracy = sapply(List.res, function(x)  x$overall[1]),
                            specif = sapply(List.res, function(x)  x$byClass[2]),
                            sens = sapply(List.res, function(x)  x$byClass[1]),
                            recall = sapply(List.res, function(x)  x$byClass[6]),
                            f1 = sapply(List.res, function(x)  x$byClass[7]))  
    list.results[[i]][[j]] <- mertics
  }
}








a <- lapply(1:7, function(i) colMeans(do.call(rbind, (lapply(list.results, function(x) x[[1]][i,])))))
res_simulation1 <- as.data.frame(do.call(rbind, lapply(a, as.vector)))
res_simulation1 <- cbind(my.var=row.names(list.results[[1]][[1]]), res_simulation1)
colnames(res_simulation1) <-  c('group',names(list.results[[1]][[1]]))


ggplot(res_simulation1, aes(x = as.character(group))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 1")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))


a <- lapply(1:21, function(i) colMeans(do.call(rbind, (lapply(list.results, function(x) x[[2]][i,])))))
res_simulation2 <- as.data.frame(do.call(rbind, lapply(a, as.vector)))
res_simulation2 <- cbind(my.var=row.names(list.results[[2]][[2]]), res_simulation2)
colnames(res_simulation2) <-  c('group',names(list.results[[2]][[2]]))


ggplot(res_simulation2, aes(x = as.character(group))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 2")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))


a <- lapply(1:35, function(i) colMeans(do.call(rbind, (lapply(list.results, function(x) x[[3]][i,])))))
res_simulation3 <- as.data.frame(do.call(rbind, lapply(a, as.vector)))
res_simulation3 <- cbind(my.var=row.names(list.results[[3]][[3]]), res_simulation3)
colnames(res_simulation3) <-  c('group',names(list.results[[3]][[3]]))


ggplot(res_simulation3, aes(x = as.character(group))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 3")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))



a <- lapply(1:21, function(i) colMeans(do.call(rbind, (lapply(list.results, function(x) x[[2]][i,])))))
res_simulation2 <- as.data.frame(do.call(rbind, lapply(a, as.vector)))
res_simulation2 <- cbind(my.var=row.names(list.results[[2]][[2]]), res_simulation2)
colnames(res_simulation2) <-  c('group',names(list.results[[2]][[2]]))


ggplot(res_simulation2, aes(x = as.character(group))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 2")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))


a <- lapply(1:35, function(i) colMeans(do.call(rbind, (lapply(list.results, function(x) x[[4]][i,])))))
res_simulation4 <- as.data.frame(do.call(rbind, lapply(a, as.vector)))
res_simulation4 <- cbind(my.var=row.names(list.results[[4]][[4]]), res_simulation4)
colnames(res_simulation4) <-  c('group',names(list.results[[4]][[4]]))


ggplot(res_simulation4, aes(x = as.character(group))) +
  geom_line(aes(y = kappa),color = 'darkred',group='kappa' ) +
  geom_point(aes(y = kappa),color = 'darkred',group='kappa' )+
  geom_line(aes(y = accuracy),color = 'red',group='accuracy' ) +
  geom_point(aes(y = accuracy),color = 'red',group='accuracy' )+
  
  geom_line(aes(y = specif),color = 'blue',group='specif' ) +
  geom_point(aes(y = specif),color = 'blue',group='specif' )+
  
  geom_line(aes(y = sens),color = 'darkblue',group='sens' ) +
  geom_point(aes(y = sens),color = 'darkblue',group='sens' )+
  
  geom_line(aes(y = recall),color = 'green',group='recall' ) +
  geom_point(aes(y = recall),color = 'green',group='recall' )+
  
  geom_line(aes(y = f1),color = 'darkgreen',group='f1' ) +
  geom_point(aes(y = f1),color = 'darkgreen',group='f1' )+
  
  labs(x = "", y = "metrics", title = "Comb 4")+
  theme(axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1))



trctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 2)
x <- 1:7


svm.c <- train(group ~., data = data_balanced_both[,c(1,3,6,8)], method = "svmLinear",
                    trControl=trctrl,
                    tuneLength = 10)
svm_Linear


test_pred <- predict(svm.c, newdata = data.test.both[,c(1,3,6)])
a <- confusionMatrix(table(test_pred, data.test.both[,8]))


kernlab::plot(summary(svm.c))




library(rgl)



plot3d(dat_total$Exploration.Proportion[1:43], dat_total$Casp[1:43], dat_total$ERA[1:43], size = 14,col=as.integer(as.factor(dat_total$group)),
       xlab = 'Exploration Proportion', ylab = "Caspase 3", zlab = "ERA")













# Decision trees

latlontree = rpart(group ~ Exploration.Proportion + Caspase.3 + ERA, data=data_balanced_both )
plot(latlontree)
text(latlontree)








# Logistic regression

#load required library
library(glmnet)
#convert training data to matrix format
x <- model.matrix(group~., data_balanced_both[,c(1,3,6,8)])
#convert class to numerical variable
y <- data_balanced_both[,8]
#perform grid search to find optimal value of lambda
#family= binomial => logistic regression, alpha=1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x,y,alpha=1,family="binomial", type.measure = "mse")
#plot result
plot(cv.out)


#min value of lambda
lambda_min <- cv.out$lambda.min
#best value of lambda
lambda_1se <- cv.out$lambda.1se
#regression coefficients
coef(cv.out,s=lambda_1se)

model <- glmnet(x, y, alpha = 1, family = "binomial",
                lambda = lambda_min)

#get test data
x_test <- model.matrix(group~.,data.test.both[,c(1,3,6,8)])
#predict class, type="class"
lasso_prob <- predict(model,newx = x_test,s=lambda_1se,type="response")
#translate probabilities to predictions
lasso_predict <- rep("0",nrow(data.test.both[,c(1,3,6,8)]))
lasso_predict[lasso_prob>.5] <- "1"

table(pred=lasso_predict,true=data.test.both[,8])

#accuracy
mean(lasso_predict==data.test.both[,8])










coef(model)
# Make predictions on the test data
probabilities <- model %>% predict(newx = x_test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- data.test.both[,8]
mean(predicted.classes == observed.classes)


roc.curve(data.test.both$group, predicted.classes)
accuracy.meas(data.test.both$group, predicted.classes)
a <- confusionMatrix(table(predicted.classes, data.test.both[,8]))




f1 <- data.frame(  kappa = rep(0,100),
                   accuracy = rep(0,100),
                   specif = rep(0,100),
                   sens = rep(0,100),
                   recall = rep(0,100),
                   f1 = rep(0,100))


list.results <- rep(list(f1), 6)
names(list.results) <- c("logistic", "knn", "decision", "lda", "qda", "svc")


for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  x <- model.matrix(group~., data_balanced_both[,c(1,3,6,8)])
  #convert class to numerical variable
  y <- data_balanced_both[,8]
  #perform grid search to find optimal value of lambda
  #family= binomial => logistic regression, alpha=1 => lasso
  # check docs to explore other type.measure options
  cv.out <- cv.glmnet(x,y,alpha=1,family="binomial", type.measure = "mse")
  #best value of lambda
  lambda_min <- cv.out$lambda.min

  model <- glmnet(x, y, alpha = 1, family = "binomial",
                  lambda = lambda_min)
  
  #get test data
  x_test <- model.matrix(group~.,data.test.both[,c(1,3,6,8)])
  #predict class, type="class"
  lasso_prob <- predict(model,newx = x_test,s=lambda_min,type="response")
  #translate probabilities to predictions
  lasso_predict <- rep("0",nrow(data.test.both[,c(1,3,6,8)]))
  lasso_predict[lasso_prob>.5] <- "1"
  # Make predictions on the test data
  probabilities <- model %>% predict(newx = x_test)
  predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
  # Model accuracy
  observed.classes <- data.test.both[,8]
  mean(predicted.classes == observed.classes)
  a <- confusionMatrix(table(predicted.classes, data.test.both[,8]))
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["logistic"]][i,] <- metrics
}





for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  
  pr <- knn(data_balanced_both[,c(1,3,6)],data.test.both[,c(1,3,6)],cl=data_balanced_both[,8],k=3)
  a <- confusionMatrix(table(pr, data.test.both[,8]))
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["knn"]][i,] <- metrics
}


joiner <- function(x){
  if(x > 0.5){
    return(1)
  } else {
    return(0)
  }
}
for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  tree.both <- rpart(group ~ ., method = 'class', data = data_balanced_both[,c(1,3,6,8)])
  pred.tree.both <- predict(tree.both, newdata = data.test.both[,c(1,3,6)])

  pred.tree.both <- as.data.frame(pred.tree.both)
  pred.tree.both$con <- sapply(pred.tree.both$`1`, joiner)
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  a <- confusionMatrix(table(pred.tree.both$con, data.test.both[,8]))
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["decision"]][i,] <- metrics
}


library(MASS)


for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  lda.fit=lda(group~., data=data_balanced_both[,c(1,3,6,8)])
  lda.pred=predict(lda.fit, data.test.both[,c(1,3,6)])
  
  a <- confusionMatrix(table(lda.pred$class, data.test.both[,8]))
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["lda"]][i,] <- metrics
}


for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  qda.fit=qda(group~., data=data_balanced_both[,c(1,3,6,8)])
  qda.pred=predict(qda.fit, data.test.both[,c(1,3,6)])
  
  a <- confusionMatrix(table(qda.pred$class, data.test.both[,8]))
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["qda"]][i,] <- metrics
}




for(i in 1:100){
  ##extract training set
  train <- dat_total_norm[1:43,]
  train$group <- dat_total[1:43,9]
  rows <- sample(nrow(train))
  train <- train[rows,]
  ##extract testing set
  test <- dat_total_norm[44:90,] 
  test$group <- dat_total[44:90,9]
  rows <- sample(nrow(test))
  test <- test[rows,]
  
  # Imputing
  data_balanced_both <- ovun.sample(group ~ ., data = train, method = "both", p=0.5, N=43)$data
  rows <- sample(nrow(data_balanced_both))
  data_balanced_both <- data_balanced_both[rows,]
  
  
  data.test.both <- ovun.sample(group ~ ., data = test, method = "both",p=0.5, N = 47)$data
  rows <- sample(nrow(data.test.both))
  data.test.both <- data.test.both[rows,]
  
  data_balanced_both$group <- as.factor(data_balanced_both$group)
  data.test.both$group <- as.factor(data.test.both$group)
  
  svm.c <- train(group ~., data = data_balanced_both[,c(1,3,6,8)], method = "svmLinear")
  
  
  test_pred <- predict(svm.c, newdata = data.test.both[,c(1,3,6)])
  a <- confusionMatrix(table(test_pred, data.test.both[,8]))
  
  metrics <- data.frame(  kappa = a$overall[2],
                          accuracy = a$overall[1],
                          specif = a$byClass[2],
                          sens = a$byClass[1],
                          recall = a$byClass[6],
                          f1 = a$byClass[7])  
  list.results[["svc"]][i,] <- metrics
}




list.results <- Map(cbind, list.results, type = names(list.results))

results <- do.call(rbind, list.results)


library(reshape2)


results_melt <- melt(results, measure.vars = 1:6)

ggplot(results_melt, aes(x = type, y = value, color = type)) +
  geom_boxplot() +
  facet_wrap(~variable)



colMeans(list.results[["svc"]][,-7])






library(GGally)

data_balanced_both$group <- as.factor(data_balanced_both$group)

ggpairs(
  data = data_balanced_both,
  columns = c(1:8),
  diag = list(continuous = wrap("barDiag", color = "blue", size =4)),
  upper = list(continuous = wrap("cor", size = 4, bins = 60))
)




write.csv(data_balanced_both, "data_train.csv")
write.csv(data.test.both, "data_test.csv")




library(rgl)
library(misc3d)
# Plot original data
plot3d(data_balanced_both[,c(1,3,6)], col=as.integer(data_balanced_both[,8]))

# Get decision values for a new data grid
newdat.list = lapply(data_balanced_both[,c(1,3,6)], function(x) seq(min(x), max(x), len=43))
newdat      = expand.grid(newdat.list)
newdat.pred = predict(svm.c, data.test.both[,c(1, 3, 6)], decision.values=T)
newdat.dv   = attr(newdat.pred, 'decision.values')
newdat.dv   = array(newdat.pred, dim=rep(43, 3))

# Fit/plot an isosurface to the decision boundary
contour3d(newdat.dv, level=0, x=newdat.list$Exploration.Proportion, y=newdat.list$Caspase.3, z=newdat.list$ERA, add=T)











w <- unlist(svm.c$finalModel@coef) %*% svm.c$finalModel@SVindex
detalization <- 100                                                                                                                                                                 
grid <- expand.grid(seq(from=min(data_balanced_both$Exploration.Proportion),to=max(data_balanced_both$Exploration.Proportion),length.out=detalization),                                                                                                         
                    seq(from=min(data_balanced_both$Caspase.3),to=max(data_balanced_both$Caspase.3)),length.out=detalization)                                                                                                         
z <- (svm.c$finalModel@b - w[1,1]*grid[,1] - w[1,2]*grid[,2]) / w[1,3]

plot3d(grid[,1],grid[,2],z)  # this will draw plane.
# adding of points to the graphics.
points3d(t$x[which(t$cl==-1)], t$y[which(t$cl==-1)], t$z[which(t$cl==-1)], col='red')
points3d(t$x[which(t$cl==1)], t$y[which(t$cl==1)], t$z[which(t$cl==1)], col='blue')





svm.c$finalModel@










require(e1071) # for svm()                                                                                                                                                          
require(rgl) # for 3d graphics.                                                                                                                                                                                    
set.seed(12345)                                                                                                                                                                     
seed <- .Random.seed                                                                                                                                                                
t <- data_balanced_both[,c(1,3,6,8)]
t2 <- rbind(data_balanced_both[,c(1,3,6,8)], data.test.both[,c(1,3,6,8)])
t2$group <- as.factor(ifelse(t2$group == 1,1,-1))
names(t2) <- c("x", "y", "z", "cl")

svm_model <- svm(cl~., t2, type='C-classification', kernel='linear',scale=FALSE)

pred <- predict(svm_model, data.test.both[,c(1,3,6,8)])

confusionMatrix(pred, data.test.both[,8])


w <- t(svm_model$coefs) %*% svm_model$SV

detalization <- 100                                                                                                                                                                 
grid <- expand.grid(seq(from=min(data_balanced_both$Exploration.Proportion),to=max(data_balanced_both$Exploration.Proportion),length.out=detalization),                                                                                                         
                    seq(from=min(data_balanced_both$Caspase.3),to=max(data_balanced_both$Caspase.3)),length.out=detalization)                                                                                                         
z <- (svm_model$rho- w[1,1]*grid[,1] - w[1,2]*grid[,2]) / w[1,3]

plot3d(grid[,1],grid[,2],z)  # this will draw plane.
# adding of points to the graphics.
points3d(data_balanced_both$Exploration.Proportion[which(data_balanced_both$group==0)], 
         data_balanced_both$Caspase.3[which(data_balanced_both$group==0)], 
         data_balanced_both$ERA[which(data_balanced_both$group==0)], col='red')
points3d(data_balanced_both$Exploration.Proportion[which(data_balanced_both$group==1)], 
         data_balanced_both$Caspase.3[which(data_balanced_both$group==1)], 
         data_balanced_both$ERA[which(data_balanced_both$group==1)], col='blue')







grid <- expand.grid(seq(from=0,to=1.2,length.out=detalization),                                                                                                         
                    seq(from=0,to=1.2,length.out=detalization))                                                                                                         
z <- (svm_model$rho- w[1,1]*grid[,1] - w[1,2]*grid[,2]) / w[1,3]

plot3d(grid[,1],grid[,2],z,xlab = 'Exploration Proportion', ylab = "Caspase 3", zlab = "ERA")  # this will draw plane.
# adding of points to the graphics.
points3d(t2$x[which(t2$cl==-1)], t2$y[which(t2$cl==-1)], t2$z[which(t2$cl==-1)], col='red', size = 14)
points3d(t2$x[which(t2$cl==1)], t2$y[which(t2$cl==1)], t2$z[which(t2$cl==1)], col='blue', size = 14)


table(t$cl)
table(data_balanced_both$group)



rgl.snapshot('3d_hyperplane', fmt = "png")
