# Genetic Algorithm and machine leaning algorithm for Enhancing Intrusion Detection System

## This is the repo for decision tree classifier optimization using genetic algorithm
Decision tree is one of the most used algorithm for IDS. The best performing model is Random forest which is am ensemble approach that combines many decision trees and pick best amongst them to find the best accuracy for threat detection.
### benefits

- reduce features
- optimize for resource constraints

```

initail columns = 43
Selected Features: [0, 2, 3, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 21, 24, 25, 28, 30, 33, 34, 36, 39, 41]
fist run = 23 features

Selected Features: [0, 2, 4, 9, 11, 12, 13, 16, 17, 18, 22, 25, 26, 28, 29, 30, 35, 37, 38, 39, 40, 41]
second run = 22 features

Accuracy: 0.8579222853087296
Classification Report:
               precision    recall  f1-score   support

           0       0.76      0.97      0.85      9711
           1       0.97      0.77      0.86     12833

    accuracy                           0.86     22544
   macro avg       0.87      0.87      0.86     22544
weighted avg       0.88      0.86      0.86     22544

Selected Features: [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 
17, 18, 19, 20, 21, 24, 28, 29, 33, 36, 40, 41]
Accuracy: 0.859208658623137
Classification Report:
               precision    recall  f1-score   support

           0       0.76      0.97      0.86      9711       
           1       0.97      0.77      0.86     12833       

    accuracy                           0.86     22544       
   macro avg       0.87      0.87      0.86     22544       
weighted avg       0.88      0.86      0.86     22544       

nsl-kdd/randomForest.py
43
gen     nevals
0       50
1       24
2       26    
3       23    
4       36    
5       37    
6       25
7       24
8       34
9       32
10      18
11      35    
12      29    
13      30    
14      29    
15      34    
16      26    
17      29    
18      32    
19      33    
20      30    
Accuracy: 0.8283356990773598
Classification Report:
               precision    recall  f1-score   support

           0       0.72      0.97      0.83      
9711
           1       0.97      0.72      0.83     12833

    accuracy                           0.83     22544
   macro avg       0.85      0.85      0.83     22544
weighted avg       0.86      0.83      0.83     22544

```

