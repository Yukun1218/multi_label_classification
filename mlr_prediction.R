library(mlr)
library(randomForestSRC)
library(rFerns)
library(dummies)
library(mlbench)
library(tidytext)

library(tidyverse)
library(RODBC)
library(randomForest)
library(dplyr)
library(janeaustenr)
library(tm)
library(stringdist)
library(glmnet)
library(nnet)

#-- model setting -- 
lrn.rfsrc = makeLearner("multilabel.randomForestSRC")
lrn.rFerns = makeLearner("multilabel.rFerns")
lrn.rFerns

lrn.br = makeLearner("classif.rpart", predict.type = "prob")
lrn.br = makeMultilabelBinaryRelevanceWrapper(lrn.br)
lrn.br

lrn.br2 = makeMultilabelBinaryRelevanceWrapper("classif.rpart")
lrn.br2

# SQL data
project_component <- read.csv('project_component.csv')

#-- word2vec process on training set ---
topics <- read.csv('project_portfolio_analysis.csv', header = T, sep=',') %>%
  select(starts_with('ID'), starts_with('Type')) %>%
  select(WB_Project_ID = "ID..link.to.ops.portal.", 
    Type_1 = "Type.of.support..1.", 
    Type_2 = "Type.of.support..2." , 
    Type_3 = "Type.of.support..3." , 
    Type_4 = "Type.of.support..4." , 
    Type_5 = "Type.of.support..5." ) %>%
  slice(2:31) %>%
  left_join(project_component, by = 'WB_Project_ID') %>%
  left_join(project_list, by = 'WB_Project_ID')

train <- topics %>%
  mutate_if(is.factor, as.character) %>%
  group_by(WB_Project_ID) %>%
  mutate(text = paste0(Project_component, WB_Project_Name, collapse=' ')) 

#-- tag list --
list <- unique(c(train$Type_1, train$Type_2, train$Type_3, train$Type_4, train$Type_5)) %>%
  as.data.frame() 
names(list) <-  'Type'  
list <- list %>%
  mutate_if(is.factor, as.character) %>%
  filter(nchar(Type)>0) %>%
  arrange(desc(Type)) %>%
  rownames_to_column() %>%
  rename(type_code = rowname)

#-- training set preparation--
topic_1 <- topics[1:6] %>%
  gather(key = 'Label_n',value = 'Type', - WB_Project_ID) %>%
  select(-Label_n) %>%
  filter(nchar(Type)>0) %>%
  left_join(list, by = 'Type') %>%
  select(-Type) %>%
  mutate(id = as.numeric(gsub("P", "", WB_Project_ID)), 
    type_code = as.numeric(type_code)) %>%
  select(id, type_code) %>%
  as.data.frame()

topic_1 <- cbind(topic_1, dummy(topic_1$type_code, sep = '_'))
names(topic_1) <- gsub("topic_1", 'label', names(topic_1))

topic_2 <- select(topic_1, - type_code) %>%
  group_by(WB_Project_ID) %>%
  summarise_if(is.numeric, sum) %>%
  mutate_at(vars(matches("label")), .funs=function(x){ifelse(as.numeric(x)==0, FALSE, TRUE)}) %>%
  select(-WB_Project_ID)

all_text <- left_join(project_component, project_list, by = "WB_Project_ID") %>%
  group_by(WB_Project_ID) %>%
  mutate(text= paste0(Project_component, WB_Project_Name, collapse = " ")) %>%
  select( -Project_component, -WB_Project_Name )

prep_fun = tolower
tok_fun = word_tokenizer

it_trainTest <- all_text$text %>% 
  prep_fun %>% tok_fun %>% 
  itoken(ids = all_text$WB_Project_ID, progressbar = FALSE)

dtm_all = create_dtm(it_trainTest, vectorizer) 

df_dtm_all <- tidy(dtm_all)%>%
  filter(nchar(column)>1) %>%
  group_by(row) %>%
  spread(key= column, value = value) %>%
  rename(WB_Project_ID = row)

df_dtm_all[is.na(df_dtm_all)] <- 0

# form training df(proj with tags)
data_train <- left_join(topic_2, df_dtm_all, by = "WB_Project_ID") %>%
  as.data.frame()

# Keep the training and testing ids
training_id <- c(topic_2$WB_Project_ID)
data_test <- filter(df_dtm_all , !(WB_Project_ID %in% training_id)) %>%
  ungroup() %>%
  as.data.frame()

# create fake columns for test data
labels = colnames(data_train)[2:30]
label_str <- data.frame(matrix(ncol = 29, nrow = nrow(data_test))) 
colnames(label_str) <- labels

# create test data set 
# data_test <- filter(df_dtm_all , !(WB_Project_ID %in% training_id))
data_test <- bind_cols(data_test, label_str) %>%
  select(WB_Project_ID, starts_with('label'), everything())

data_all <- bind_rows(data_train, data_test) %>%
  select(-WB_Project_ID)

names(data_all) <- make.names(names(data_all))
  
labels = colnames(data_all)[1:29]
data.task <- makeMultilabelTask(id = "multi", data = data_all[1:30,], target = labels)
# data.task
mod = train(lrn.rfsrc, data.task)
pred = predict(mod, newdata = data_all[31:150,] )

revert_data <- bind_rows(data_train, data_test)
proj_id <- revert_data[31:150, 'WB_Project_ID']

results <- data.frame(proj_id, pred$data) %>%
  select(WB_Project_ID = proj_id, starts_with('response')) %>%
  gather(key=type, value = value, -WB_Project_ID) %>%
  group_by(WB_Project_ID) %>%
  filter(value==TRUE) %>%
  mutate(type = gsub("response.label_", "", type)) %>%
  mutate(type_code = as.character(type)) %>%
  left_join(list, by = 'type_code') %>%
  select(WB_Project_ID, Type)
