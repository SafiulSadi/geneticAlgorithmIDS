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
```