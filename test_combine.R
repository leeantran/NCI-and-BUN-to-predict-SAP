library(caret)
library(pROC)
library(gam)     # Thu vien ho tro GAM trong caret
library(rpart)   # Thu vien Decision Tree
library(officer)
library(flextable)

# 1. Chuan bi du lieu
set.seed(234)

# Chuyen doi bien dich thanh factor hop le cho caret
# Luu y: caret thich label hop le (khong bat dau bang so)
train$sap <- factor(train$sap, levels = c(0, 1), labels = c("nonsap", "sap"))
test$sap <- factor(test$sap, levels = c(0, 1), labels = c("nonsap", "sap"))

# Danh sach bien
vars <- c("nci", "neu_crp", "plr", "nlr", "bun", "lym", 
          "wbc", "neu", "hct", "cre", "crp", "plt", "cpr")

# Tao to hop bien
combinations <- combn(vars, 2)

# --- THIET LAP 10-FOLD CV ---
# Day la phan ban hoi: Cau hinh de train mo hinh voi 10-fold CV
cv_ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary, # De toi uu hoa theo ROC
  savePredictions = "final",
  verboseIter = FALSE
)

# Khoi tao bang ket qua
results <- data.frame(
  Var1 = character(),
  Var2 = character(),
  AUC_Combine_Tree = numeric(), 
  AUC_V1_GAM = numeric(),    
  AUC_V2_GAM = numeric(),    
  P_val_vs_V1 = numeric(), 
  P_val_vs_V2 = numeric(), 
  stringsAsFactors = FALSE
)

print(paste("Tong so cap bien can chay:", ncol(combinations)))
print("Dang chay training voi 10-fold CV... (Se lau hon code cu mot chut)")

# 2. Vong lap chay mo hinh
for (i in 1:ncol(combinations)) {
  var1 <- combinations[1, i]
  var2 <- combinations[2, i]
  
  tryCatch({
    # --- MO HINH 1: GAM Don le Var1 (Co 10-fold CV) ---
    # method = "gam" trong caret su dung splines tu dong
    m1 <- train(as.formula(paste("sap ~", var1)), 
                data = train, 
                method = "gam", 
                metric = "ROC",
                trControl = cv_ctrl)
    
    p1 <- predict(m1, test, type = "prob")[, "sap"]
    roc1 <- roc(test$sap, p1, quiet = TRUE)
    
    # --- MO HINH 2: GAM Don le Var2 (Co 10-fold CV) ---
    m2 <- train(as.formula(paste("sap ~", var2)), 
                data = train, 
                method = "gam", 
                metric = "ROC",
                trControl = cv_ctrl)
    
    p2 <- predict(m2, test, type = "prob")[, "sap"]
    roc2 <- roc(test$sap, p2, quiet = TRUE)
    
    # --- MO HINH 3: Decision Tree Ket hop (Co 10-fold CV) ---
    # method = "rpart" se tu dong tune tham so cp (complexity parameter)
    m3 <- train(as.formula(paste("sap ~", var1, "+", var2)), 
                data = train, 
                method = "rpart", 
                metric = "ROC",
                trControl = cv_ctrl)
    
    p3 <- predict(m3, test, type = "prob")[, "sap"]
    roc3 <- roc(test$sap, p3, quiet = TRUE)
    
    # --- SO SANH THONG KE (DeLong Test tren tap Test doc lap) ---
    test_vs_v1 <- roc.test(roc3, roc1, method = "delong")
    test_vs_v2 <- roc.test(roc3, roc2, method = "delong")
    
    # Luu ket qua
    results[i, ] <- list(
      var1, 
      var2, 
      as.numeric(roc3$auc), 
      as.numeric(roc1$auc), 
      as.numeric(roc2$auc), 
      test_vs_v1$p.value, 
      test_vs_v2$p.value
    )
    
  }, error = function(e) {
    message(paste("Loi tai cap:", var1, "-", var2, ":", e$message))
  })
}

# 3. Xu ly ket qua va Xuat file Word
results <- na.omit(results)

if (nrow(results) > 0) {
  
  # Sap xep theo AUC cua mo hinh Tree ket hop
  results_sorted <- results[order(-results$AUC_Combine_Tree), ]
  
  # Lam tron so
  cols_num <- c("AUC_Combine_Tree", "AUC_V1_GAM", "AUC_V2_GAM", "P_val_vs_V1", "P_val_vs_V2")
  results_sorted[cols_num] <- lapply(results_sorted[cols_num], as.numeric)
  results_sorted[cols_num] <- round(results_sorted[cols_num], 4)
  
  # Tao bang dep bang flextable
  ft <- flextable(results_sorted)
  ft <- theme_vanilla(ft)
  ft <- autofit(ft)
  ft <- bold(ft, part = "header")
  # To dam cac dong co y nghia thong ke (P < 0.05)
  ft <- color(ft, i = ~ P_val_vs_V1 < 0.05 & P_val_vs_V2 < 0.05, color = "red")
  
  # Xuat ra Word
  doc <- read_docx()
  doc <- body_add_par(doc, value = "Ket qua so sanh: Decision Tree (2 bien) vs GAM (1 bien)", style = "heading 1")
  doc <- body_add_par(doc, value = "Ghi chu: Cac mo hinh da duoc huan luyen voi 10-fold Cross-Validation.", style = "normal")
  doc <- body_add_flextable(doc, value = ft)
  
  file_name <- "Ket_qua_CV_Tree_vs_GAM.docx"
  print(doc, target = file_name)
  
  message(paste("Da xuat file Word thanh cong:", file.path(getwd(), file_name)))
  print(head(results_sorted, 10))
  
} else {
  message("Khong co ket qua nao de xuat (Dataframe rong).")
}