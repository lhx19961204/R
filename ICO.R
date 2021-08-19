#导入R包
library(missForest)
library(caret)
library(Metrics)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(readr)
library(corrgram)

#设置工作目录
setwd("F:/Machine learning/Assessment/")

###一、数据读取与预处理

#读取数据文件
dat <- read.csv("dataset ICO.csv", encoding = 'UTF-8')
#删除数据id列
dat <- dat[, -1]
#查看数据维度
dim(dat)
#查看数据属性
str(dat)
#这个地方给你替换成readr里面的parser_number解析数字
dat$priceETH <- parse_number(as.character(dat$priceETH))
str(dat)
levels(dat$country)

#筛选数值型变量,因变量是success，使用的自变量：tokenNum，teamSize，USA, UK, othercountry，overallrating，offered_ownership，duration，priceETH，isERC，softcap，hardcap, whitepaper, video, socialMedia
dat <- dat[, c(1,2,3,4,5,6,9,10,13,16,17,21,22,23,24,25)]

#对tokenNum和teamSize做对数转换
dat$tokenNum <- log(dat$tokenNum)
dat$teamSize <- log(dat$teamSize)

#观测数据离群值
boxplot(dat)
#将离群值替换成NA
dat$tokenNum[dat$tokenNum > boxplot.stats(dat$tokenNum)$stats[5]] <- NA
dat$tokenNum[dat$tokenNum < boxplot.stats(dat$tokenNum)$stats[1]] <- NA
dat$teamSize[dat$teamSize < boxplot.stats(dat$teamSize)$stats[1]] <- NA
dat$offered_ownership[dat$offered_ownership > boxplot.stats(dat$offered_ownership)$stats[5]] <- NA
dat$offered_ownership[dat$offered_ownership < boxplot.stats(dat$offered_ownership)$stats[1]] <- NA
dat$duration[dat$duration > boxplot.stats(dat$duration)$stats[5]] <- NA
dat$duration[dat$duration < boxplot.stats(dat$duration)$stats[1]] <- NA
dat$priceETH[dat$priceETH > boxplot.stats(dat$priceETH)$stats[5]] <- NA

#统计含有缺失值的数据行
dim(dat[!complete.cases(dat),])
#共有665个观测含有缺失值
#统计每个变量含有缺失值的个数
apply(dat, 2, function(x){sum(is.na(x))})
#将分类变量转换成因子类型
dat$USA <- as.factor(dat$USA)
dat$UK <- as.factor(dat$UK)
dat$othercountry <- as.factor(dat$othercountry)
dat$success <- as.factor(dat$success)
dat$softcap <- as.factor(dat$softcap)
dat$hardcap <- as.factor(dat$hardcap)
dat$whitepaper <- as.factor(dat$whitepaper)
dat$video <- as.factor(dat$video)
dat$socialMedia <- as.factor(dat$socialMedia)

#利用随机森林方法对缺失值进行插补
set.seed(123)
dat_all <- missForest(dat)$ximp
boxplot(dat_all)
###二、调查目标变量与其他变量的关系

#用corrgram对因变量和自变量分析相关关系
dat_all[sapply(dat, is.factor)] <- lapply(dat_all[sapply(dat_all, is.factor)], function(x){as.numeric(as.character(x))})
corrgram(dat_all, order = TRUE, lower.panel = panel.shade, upper.panel = panel.pie,text.panel = panel.txt,
         main = "Corrgram of ICO intercorrelations")

#由此可见success与overallrating、teamSize、socialMedia、video和tokenNum的相关性最强
#利用逻辑回归分析判断success与各变量的相关性
summary(step(glm(as.factor(success) ~., dat_all, family = "binomial")))
#由此可见success与overallrating、socialMedia、whitepaper、video、tokenNum、teamSize和isERC的相关性最强
dat_all <- dat_all[, -c(5,6,12)]
### 三、建模与调参
#将数据集切分为80%的训练集和20%的测试集
set.seed(123)
index <- createDataPartition(dat_all$success, p = 0.8)
train_dat <- dat_all[index$Resample1, ]
test_dat <- dat_all[-index$Resample1,]
#(1)knn
#使用knn3函数来建立KNN分类器，使用参数K=5来指定近邻数的取值为5.
knn_model <- knn3(as.factor(success) ~., train_dat, k = 15)
#使用predict函数来预测测试集的类别
test_pre <- predict(knn_model, test_dat, type = "class")
#计算预测的准确性
accuracy(as.factor(test_dat$success), test_pre)#使用混淆矩阵热力图可视化哪些预测正确哪些不正确
confusionMatrix(as.factor(test_dat$success), test_pre)
# 绘制ROC曲线
knn_roc <- roc(test_dat$success, as.numeric(test_pre))
knn_auc <- as.numeric(unlist(strsplit(as.character(auc(knn_roc)), " "))[1])
plot(knn_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE, auc.polygon.col = "skyblue",
     print.thres = TRUE, main = 'KNN模型ROC曲线')

#通过交叉验证的方法，搜索KNN分类中合适的近邻数量
set.seed(123)
#使用5折交叉验证
trcl <- trainControl(method = "cv", number = 5)
#expand.grid函数用于指定需要搜索的模型参数（k值）
trgrid <- expand.grid(k = seq(1, 25, 2))
#测试不同k值下的分类器效果
knnFit <- train(as.factor(success) ~., dat_all, method = "knn", trControl = trcl, tuneGrid = trgrid)
#绘制不同k下的KNN分类器的精确度图
plot(knnFit, main = "KNN", family = "STKaiti")
#从图中可以看出，随着近邻数K的增加，分类器的分类效果在缓慢增加，当近邻数等于15时分类效果最好。
#估计变量重要性
knn_importance <- varImp(knnFit, scale = FALSE)
print(knn_importance)
plot(knn_importance, main = "Importance of KNN Variables")

#(2)决策树
#使用rpart函数来建立决策树分类器，使用参数cp=0.000001指定决策树模型在剪枝时采用的阈值。
tree_model <- rpart(as.factor(success) ~., train_dat, cp = 0.01)
#使用raprt.plot函数将建立的决策树进行可视化
rpart.plot(tree_model, type = 2, extra = "auto", under = T, fallen.leaves = F, cex = 0.7, main = "Decision tree")
#使用predict函数来预测测试集的类别
test_pre <- predict(tree_model, test_dat, type = "class")
#计算预测的准确性
accuracy(as.factor(test_dat$success), test_pre)
#使用混淆矩阵热力图可视化哪些预测正确哪些不正确
confusionMatrix(as.factor(test_dat$success), test_pre)


# 绘制ROC曲线
tree_roc <- roc(test_dat$success, as.numeric(test_pre))
tree_auc <- as.numeric(unlist(strsplit(as.character(auc(tree_roc)), " "))[1])
plot(tree_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE, auc.polygon.col = "skyblue",
     print.thres = TRUE, main = '决策树模型ROC曲线')
#通过交叉验证的方法，搜索决策树分类中合适的cp值
set.seed(123)
#使用5折交叉验证
trcl <- trainControl(method = "cv", number = 5)
#expand.grid函数用于指定需要搜索的模型参数（cp值）
trgrid <- expand.grid(cp = seq(0, 0.25, 0.01))
#测试不同cp值下的分类器效果
treeFit <- train(as.factor(success) ~., dat_all, method = "rpart", trControl = trcl, tuneGrid = trgrid)
#绘制不同cp下的决策树分类器的精确度图
plot(treeFit, main = "Decision tree", family = "STKaiti")
#从图中可以看出，随着cp的增加，分类器的分类效果在降低，当近cp等于0.05时分类效果最好。
#估计变量重要性
tree_importance <- varImp(treeFit, scale = FALSE)
print(tree_importance)
plot(tree_importance, main = "Importance of DT Variables")

#(3)随机森林
#使用randomForest函数来建立随机森林分类器，使用参数ntree=200表示使用200棵决策树用于随机森林的分类。
#mtry在randomForest函数里面调整
rf_model <- randomForest(as.factor(success) ~., train_dat, mtry = 2,
                         node_size = 11, samp_size = 0.65,
                         ntree = 3000, proximity = T)
plot(rf_model)

params <- expand.grid(
        # the max value should be equal to number of predictors
        mtry = c(1:12),
        # the node sizes
        node_size = seq(3, 15, by = 2),
        # the within training data sample split
        samp_size = c(.65, 0.7, 0.8)
)
head(params)
rf.grid = vector()
# now run the loop
for(i in 1:nrow(params)) {
  # create the model
  rf.i <- ranger(
    formula = as.factor(success) ~.,
    data = train_dat,
    num.trees = 200,
    mtry = params$mtry[i],
    min.node.size = params$node_size[i],
    sample.fraction = params$samp_size[i],
    seed = 123
 )
 # add OOB error to rf.grid
 rf.grid <- c(rf.grid, sqrt(rf.i$prediction.error))
 # print to see progress
 if (i%%10 == 0) cat(i, "\t") 
}
plot(rf.grid)
# add the result to the params grid
params$mtry
params$node_size
params$OOB = rf.grid

# The results can be inspected and the best performing combination of parameters extracted using which.min:
params[which.min(params$OOB),]

#随机森林分类器的变量重要性可视化图
varImpPlot(rf_model, pch = 20, main = "Importance of RF Variables")
#使用predict函数来预测测试集的类别
test_pre <- predict(rf_model, test_dat, type = "class")

#计算预测的准确性
accuracy(as.factor(test_dat$success), test_pre)
#使用混淆矩阵热力图可视化哪些预测正确哪些不正确
confusionMatrix(as.factor(test_dat$success), test_pre)
# 绘制ROC曲线
rf_roc <- roc(test_dat$success, as.numeric(test_pre))
rf_auc <- as.numeric(unlist(strsplit(as.character(auc(rf_roc)), " "))[1])
plot(rf_roc, print.auc = TRUE, auc.polygon = TRUE, legacy.axes=TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE, auc.polygon.col = "skyblue",
     print.thres = TRUE, main = '随机森林模型ROC曲线')

#通过交叉验证的方法，搜索随机森林分类中合适的mtry节点值，可确定每次迭代的变量抽样数值，用于二叉树的变量个数
set.seed(123)
#使用5折交叉验证
trcl <- trainControl(method = "cv", number = 5)
#expand.grid函数用于指定需要搜索的模型参数（mtry值）
trgrid <- expand.grid(mtry = seq(2, 12, 1))
#测试不同mtry值下的分类器效果
rfFit <- train(as.factor(success) ~., dat_all, method = "rf", trControl = trcl, tuneGrid = trgrid)
#绘制不同mtry下的决策树分类器的精确度图
plot(rfFit, main = "RF", family = "STKaiti")
#从图中可以看出，随着mtry的增加，分类器的分类效果在减少，当近mtry等于3时分类效果最好。
#估计变量重要性
rf_importance <- varImp(rfFit, scale = FALSE)
print(rf_importance)
plot(rf_importance, main = "Importance of RF Variables")

#输出结果AUC
outinfo <- data.frame(auc = c(knn_auc, tree_auc, rf_auc), row.names = c("KNN", "决策树", "随机森林"))
t(outinfo)

#绘制ROC曲线
png("2ROC_KNN_TREE_RF.png", width = 900, height = 900,  res = 72*1.5)
plot(knn_roc, percent = TRUE, col = "red", xlim = c(1,0))
lines.roc(tree_roc, percent = TRUE, col = "blue", xlim = c(1,0))
lines.roc(rf_roc, percent = TRUE, col = "green", xlim = c(1,0))
legend("bottomright", legend = c("KNN", "DT", "RF"), col=c("red", "blue", "green"), lwd = 2)
dev.off()

plot(knn_roc, tree_roc, rf_roc, print.auc = TRUE, auc.polygon = TRUE, legacy.axes=TRUE, grid = c(0.1, 0.2),
     grid.col = c("green", "red"),
     print.thres = TRUE, main = '随机森林模型ROC曲线')
summary(dat_all
        )
