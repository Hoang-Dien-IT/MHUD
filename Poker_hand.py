import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

time_start = time.time()
data = pd.read_csv("poker-hand.csv", header=None)
data.columns = ['Chất lá 1', 'Số lá 1', 'Chất lá 2', 'Số lá 2', 'Chất lá 3', 'Số lá 3', 'Chất lá 4', 'Số lá 4', 'Chất lá 5', 'Số lá 5', 'Nhãn']
data_X = data.iloc[:, :-1]
data_y = data.iloc[:, -1]

data.hist(figsize=(20, 10))
plt.show()

kf = KFold(n_splits=10, shuffle=True)

train_size = 0
test_size = 0
count = 0

for train_index, test_index in kf.split(data_X):
    train_size = train_size + len(train_index)
    test_size = test_size + len(test_index)
    count = count + 1

with open("results.txt", "w") as file:
    file.write("===============================RESULTS=====================================\n\n")
    file.write(f"Number of training samples is: {train_size // count}\n")
    file.write(f"Number of testing samples is: {test_size // count}\n")
    file.write(f"Types of labels:{np.unique(data_y)}\n")
    file.write(f"The number of each type of label in the data set:\n{data_y.value_counts()}\n")

time_avg_knn = 0
time_avg_tree = 0
time_avg_rd = 0
fc_avg_knn = 0
fc_avg_tree = 0
fc_avg_rd = 0
rd_features_importance = 0
cnt = 0

for train_idx, test_idx in kf.split(data_X):
    with open("results.txt", "a") as file:
        file.write(f"Lần lặp thứ {cnt + 1}: \n")
    #Data division
    data_train_X, data_test_X = data_X.iloc[train_idx, ], data_X.iloc[test_idx, ]
    data_train_y, data_test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]

    #KNN
    start_knn = time.time()
    Model_KNN = KNeighborsClassifier(n_neighbors=101, n_jobs=4)
    Model_KNN.fit(data_train_X, data_train_y)
    end_knn = time.time()
    data_pred_knn = Model_KNN.predict(data_test_X)
    rate_knn_fc = round(f1_score(data_test_y, data_pred_knn, average='micro') * 100, 2)
    fc_avg_knn += rate_knn_fc
    time_avg_knn += (end_knn - start_knn)

    #TREE
    start_tree = time.time()
    Model_Tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=2)
    Model_Tree.fit(data_train_X, data_train_y)
    end_tree = time.time()
    data_pred_tree = Model_Tree.predict(data_test_X, check_input=True)
    rate_tree_fc = round(f1_score(data_test_y, data_pred_tree, average='micro') * 100, 2)
    fc_avg_tree += rate_tree_fc
    time_avg_tree += (end_tree - start_tree)

    #RANDOM FOREST
    start_rd = time.time()
    Model_RD = RandomForestClassifier(criterion="entropy")
    Model_RD.fit(data_train_X, data_train_y)
    end_rd = time.time()
    data_pred_rd = Model_RD.predict(data_test_X)
    rate_rd_fc = round(f1_score(data_test_y, data_pred_rd, average='micro') * 100, 2)
    fc_avg_rd += rate_rd_fc
    time_avg_rd += (end_rd - start_rd)

    with open("results.txt", "a") as file:
        file.write(f"================================{cnt+1} st results==============================\n")
        file.write(f"F1-score for iteration {cnt + 1} of KNN is: {rate_knn_fc}%\n")
        file.write(f"F1-score for iteration {cnt + 1} of Tree is: {rate_tree_fc}%\n")
        file.write(f"F1-score for iteration {cnt + 1} of Random Forest is: {rate_rd_fc}%\n")
        file.write(f"Time training for iteration {cnt + 1} of KNN is: {round((end_knn - start_knn)/60,2)} minutes.\n")
        file.write(f"Time training for iteration {cnt + 1} of DecistionTree is: {round((end_tree - start_tree)/60,2)} minutes.\n")
        file.write(f"Time training for iteration {cnt + 1} of RandomForest is: {round((end_rd - start_rd)/60,2)} minutes.\n")
        file.write(f"-------------------------")

    rd_features_importance += Model_RD.feature_importances_
    cnt += 1

with open("results.txt", "a") as file:
    file.write("======================Average results of 10 iterations=====================\n")
    file.write(f"Overall F1-score for 10 iterations of KNN:: {round(fc_avg_knn / cnt,2)}%\n")
    file.write(f"Overall F1-score for 10 iterations of Tree: {round(fc_avg_tree / cnt,2)}%\n")
    file.write(f"Overall F1-score for 10 iterations of Random Forest: {round(fc_avg_rd / cnt,2)}%\n")
    execution_knn = time_avg_knn / cnt
    execution_tree = time_avg_tree / cnt
    execution_rd = time_avg_rd / cnt
    file.write(f"Average execution time of KNN is: {round(execution_knn / 60, 2)} minutes.\n")
    file.write(f"Average execution time of DecistionTree is: {round(execution_tree / 60, 2)} minutes.\n")
    file.write(f"Average execution time of RandomForest is: {round(execution_rd / 60, 2)} minutues.\n")

# Biểu đồ dữ liệu
value_counts_train = data_y.value_counts()
label_name_train = []
number_of_labels_train = []
j = 0

for value, count in value_counts_train.items():
    label_name_train.append(j)
    number_of_labels_train.append(count)
    j += 1

colors = ['blue', 'orange', 'green','brown','bisque','antiquewhite','aqua','yellow','teal']
plt.bar(label_name_train, number_of_labels_train, color=colors)
plt.title('Number of each type of dataset label')
plt.xlabel('Label type')
plt.ylabel('Label quantity')
for i, value in enumerate(number_of_labels_train):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.show()

# Accuracy chart
categories = ['KNN', 'TREE', 'RANDOM FOREST']
values = [round(fc_avg_knn / cnt,2),round(fc_avg_tree / cnt,2), round(fc_avg_rd / cnt,2)]
colors = ['blue', 'orange', 'green']

plt.bar(categories, values,color=colors)
plt.title('Accuracy F1 score of 3 algorithms')

plt.xlabel('Algorithm name')
plt.ylabel('Precision value')

for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()

#Time chart determines time
categories = ['KNN', 'TREE', 'RANDOM FOREST']
values = [round(execution_knn / 60, 2),round(execution_tree / 60, 2), round(execution_rd / 60, 2)]

plt.bar(categories, values,color=colors)
plt.title('Running time of each model')

plt.xlabel('Algorithm name')
plt.ylabel('Run time(minutues)')

for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.show()

# Important attribute chart
feature_names = ['Chất lá 1', 'Số lá 1', 'Chất lá 2', 'Số lá 2', 'Chất lá 3', 'Số lá 3', 'Chất lá 4', 'Số lá 4', 'Chất lá 5', 'Số lá 5']
plt.figure(figsize=(10, 6))

colors = ['blue','blue', 'orange','orange', 'green','green','brown','brown','bisque','bisque']
plt.barh(feature_names, rd_features_importance / cnt,color=colors)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

#Cây => Luật
tree_rules = export_text(Model_Tree, feature_names=feature_names)
with open("results.txt", "a") as file:
    file.write(f"==========================Decision tree diagram===========================\n")
    file.write(tree_rules)

def extract_rules(tree, feature_names):
    rules = []
    stack = [(0, -1)]  # Dùng một ngăn xếp để theo dõi node và depth
    while stack:
        node, depth = stack.pop()
        if node < 0:
            continue
        if depth > 0:
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                value = np.argmax(tree.tree_.value[node])
                rule = "THEN Class {} (leaf node)".format(value)
            else:
                rule = "IF {} <= {}".format(feature_names[tree.tree_.feature[node]], tree.tree_.threshold[node])
            rules.append("{}{}".format("  " * depth, rule))
        stack.append((tree.tree_.children_left[node], depth + 1))
        stack.append((tree.tree_.children_right[node], depth + 1))
    return rules

tree_rules = extract_rules(Model_Tree, feature_names)

# In ra các luật
with open("results.txt", "a") as file:
    file.write(f"==========================Extracted Decision Tree Rules===========================\n")
    for rule in tree_rules:
        file.write(rule + "\n")

