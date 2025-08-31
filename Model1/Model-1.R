# 加载必要的库
library(rms)
library(pROC)
library(rmda)
library(caret)      # 用于计算分类指标
library(openxlsx)   # 用于写入Excel文件

# 读取数据
train <- read.csv("E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/train.csv")
val <- read.csv("E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/intra-val.csv")
test <- read.csv("E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/inter-val.csv")

# 设置datadist
dd <- datadist(train)
options(datadist="dd")

# 使用lrm函数拟合逻辑回归模型
f1 <- lrm(DM ~ T_staging
          + N_staging
          + Clinical_staging
          + GPH
          + NC
          + RATH
          + CLN
          + PF
          + NPF
          + Enhancement
          + macrocalcification
          + LNM,
          data = train, x = TRUE, y = TRUE)

# 创建列线图
nom <- nomogram(f1, fun = plogis, fun.at = c(.001, .01, seq(.1, .9, by = .4)), lp = FALSE, funlabel = "DM Risk")
plot(nom)

# 使用glm函数拟合模型用于预测
f2 <- glm(DM ~ T_staging
          + N_staging
          + Clinical_staging
          + GPH
          + NC
          + RATH
          + CLN
          + PF
          + NPF
          + Enhancement
          + macrocalcification
          + LNM,
          data = train, family = "binomial")

# 预测训练集的概率并保存
pre_train <- predict(f2, type = 'response')
train_predictions <- data.frame(ID = 1:nrow(train), Actual = train$DM, Predicted_Prob = pre_train)
write.xlsx(train_predictions, file = "E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/train_predictions.xlsx", overwrite = TRUE)

# 预测验证集的概率并保存
pre_val <- predict(f2, newdata = val, type = 'response')
val_predictions <- data.frame(ID = 1:nrow(val), Actual = val$DM, Predicted_Prob = pre_val)
write.xlsx(val_predictions, file = "E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/intra_val_predictions.xlsx", overwrite = TRUE)

# 预测测试集的概率并保存
pre_test <- predict(f2, newdata = test, type = 'response')
test_predictions <- data.frame(ID = 1:nrow(test), Actual = test$DM, Predicted_Prob = pre_test)
write.xlsx(test_predictions, file = "E:/Experiments/YulanPeng/Wenwen/2023Run/DM/Experiments/Clinical/Figs-New/inter_val_predictions.xlsx", overwrite = TRUE)

# 定义计算分类指标的函数
compute_metrics <- function(actual, predicted_probs, threshold = 0.5) {
  predicted_class <- ifelse(predicted_probs >= threshold, 1, 0)
  confusion <- confusionMatrix(as.factor(predicted_class), as.factor(actual), positive = "1")
  metrics <- data.frame(
    Accuracy = confusion$overall['Accuracy'],
    Precision = confusion$byClass['Precision'],
    Recall = confusion$byClass['Recall'],
    F1_Score = confusion$byClass['F1'],
    Sensitivity = confusion$byClass['Sensitivity'],
    Specificity = confusion$byClass['Specificity']
  )
  return(metrics)
}

# 计算训练集上的分类指标和AUC的95%置信区间
metrics_train <- compute_metrics(train$DM, pre_train)
roc_train <- roc(train$DM, pre_train)
auc_train <- auc(roc_train)
ci_train <- ci.auc(roc_train)

# 计算验证集上的分类指标和AUC的95%置信区间
metrics_val <- compute_metrics(val$DM, pre_val)
roc_val <- roc(val$DM, pre_val)
auc_val <- auc(roc_val)
ci_val <- ci.auc(roc_val)

# 计算测试集上的分类指标和AUC的95%置信区间
metrics_test <- compute_metrics(test$DM, pre_test)
roc_test <- roc(test$DM, pre_test)
auc_test <- auc(roc_test)
ci_test <- ci.auc(roc_test)

# 打印结果
print("训练集分类指标：")
print(metrics_train)
print(paste0("AUC: ", round(auc_train, 3), " (95% CI: ", round(ci_train[1], 3), "-", round(ci_train[3], 3), ")"))

print("验证集分类指标：")
print(metrics_val)
print(paste0("AUC: ", round(auc_val, 3), " (95% CI: ", round(ci_val[1], 3), "-", round(ci_val[3], 3), ")"))

print("测试集分类指标：")
print(metrics_test)
print(paste0("AUC: ", round(auc_test, 3), " (95% CI: ", round(ci_test[1], 3), "-", round(ci_test[3], 3), ")"))

# 绘制训练集的ROC曲线
plot.roc(train$DM, pre_train,
         percent = TRUE,
         print.auc = TRUE,
         ci = TRUE, ci.type = "bars",
         of = "thresholds",
         thresholds = "best",
         print.thres = "best",
         col = "blue",
         legacy.axes = TRUE,
         print.auc.x = 50,
         print.auc.y = 50
)

# 绘制验证集的ROC曲线
plot.roc(val$DM, pre_val,
         percent = TRUE,
         print.auc = TRUE,
         ci = TRUE, ci.type = "bars",
         of = "thresholds",
         thresholds = "best",
         print.thres = "best",
         col = "blue",
         legacy.axes = TRUE,
         print.auc.x = 50,
         print.auc.y = 50
)

# 绘制测试集的ROC曲线
plot.roc(test$DM, pre_test,
         percent = TRUE,
         print.auc = TRUE,
         ci = TRUE, ci.type = "bars",
         of = "thresholds",
         thresholds = "best",
         print.thres = "best",
         col = "blue",
         legacy.axes = TRUE,
         print.auc.x = 50,
         print.auc.y = 50
)

# 绘制校准曲线
# 训练集
cal_train <- calibrate(f1, method = "boot", B = 1000)
plot(cal_train, xlab = "Predicted DM", ylab = "Actual DM")

# 验证集
f3 <- lrm(val$DM ~ pre_val, x = TRUE, y = TRUE)
cal_val <- calibrate(f3, method = "boot", B = 1000)
plot(cal_val, xlab = "Predicted DM", ylab = "Actual DM")

# 测试集
f4 <- lrm(test$DM ~ pre_test, x = TRUE, y = TRUE)
cal_test <- calibrate(f4, method = "boot", B = 1000)
plot(cal_test, xlab = "Predicted DM", ylab = "Actual DM")

# 绘制决策曲线
# 训练集
Score_train <- decision_curve(DM ~ T_staging
                              + N_staging
                              + Clinical_staging
                              + GPH
                              + NC
                              + RATH
                              + CLN
                              + PF
                              + NPF
                              + Enhancement
                              + macrocalcification
                              + LNM,
                              data = train,
                              family = binomial(link = 'logit'),
                              thresholds = seq(0, 1, by = 0.01),
                              confidence.intervals = 0.95,
                              study.design = 'case-control',
                              population.prevalence = 0.3)

plot_decision_curve(list(Score_train),
                    curve.names = c('Nomogram'),
                    cost.benefit.axis = FALSE,
                    col = c('blue'),
                    confidence.intervals = FALSE,
                    standardize = FALSE,
                    legend.position = "bottomleft"
)

# 验证集
Score_val <- decision_curve(DM ~ T_staging
                            + N_staging
                            + Clinical_staging
                            + GPH
                            + NC
                            + RATH
                            + CLN
                            + PF
                            + NPF
                            + Enhancement
                            + macrocalcification
                            + LNM,
                            data = val,
                            family = binomial(link = 'logit'),
                            thresholds = seq(0, 1, by = 0.01),
                            confidence.intervals = 0.95,
                            study.design = 'case-control',
                            population.prevalence = 0.3)

plot_decision_curve(list(Score_val),
                    curve.names = c('Nomogram'),
                    cost.benefit.axis = FALSE,
                    col = c('blue'),
                    confidence.intervals = FALSE,
                    standardize = FALSE,
                    legend.position = "bottomleft"
)

# 测试集
Score_test <- decision_curve(DM ~ T_staging
                             + N_staging
                             + Clinical_staging
                             + GPH
                             + NC
                             + RATH
                             + CLN
                             + PF
                             + NPF
                             + Enhancement
                             + macrocalcification
                             + LNM,
                             data = test,
                             family = binomial(link = 'logit'),
                             thresholds = seq(0, 1, by = 0.01),
                             confidence.intervals = 0.95,
                             study.design = 'case-control',
                             population.prevalence = 0.3)

plot_decision_curve(list(Score_test),
                    curve.names = c('Nomogram'),
                    cost.benefit.axis = FALSE,
                    col = c('blue'),
                    confidence.intervals = FALSE,
                    standardize = FALSE,
                    legend.position = "bottomleft"
)

