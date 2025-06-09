## 不使用UCB1

Full valid score (0): 0.6599597585513078 w/o UCB1, average step 40.07657657657658

这非常奇怪，为什么不使用ucb1，而且没有历史的模型能够自主行动？

## 使用ucb1

* BEST MODELS: [4, 4, 2]
* valid: [0.93158953722334, 0.9336016096579477, 0.9356136820925554], avg = 0.9336016096579477
* steps: [14.72972972972973, 13.76126126126126, 14.86936936936937], avg = 
* test: [0.8943746259724715, 0.8674446439257929, 0.8782166367444644], avg = 0.8800119688809097
* test steps: [22.44747081712062, 19.215953307392997, 23.14396887159533]

```log
ERROR:model_theirs:Full valid score (0) with UCB1: 0.9154929577464789, average step 17.09009009009009
ERROR:model_theirs:Best model (0) at epoch 0, score 0.9154929577464789, average step 17.09009009009009. BEST_MODELS: [0, 0, 0]
ERROR:model_theirs:Full test score (0) with UCB1: 0.8204667863554758, average step 29.14396887159533
ERROR:model_theirs:Full valid score (0) with UCB1: 0.9044265593561368, average step 14.986486486486486
ERROR:model_theirs:Full valid score (0) with UCB1: 0.9014084507042254, average step 13.91891891891892
ERROR:model_theirs:Full valid score (0) with UCB1: 0.9124748490945674, average step 14.414414414414415
ERROR:model_theirs:Full valid score (0) with UCB1: 0.93158953722334, average step 14.72972972972973
ERROR:model_theirs:Best model (0) at epoch 4, score 0.93158953722334, average step 14.72972972972973. BEST_MODELS: [4, 0, 0]
ERROR:model_theirs:Full test score (0) with UCB1: 0.8943746259724715, average step 22.186770428015564
ERROR:model_theirs:Full valid score (1) with UCB1: 0.8631790744466801, average step 15.81081081081081
ERROR:model_theirs:Best model (1) at epoch 0, score 0.8631790744466801, average step 15.81081081081081. BEST_MODELS: [4, 0, 0]
ERROR:model_theirs:Full test score (1) with UCB1: 0.75344105326152, average step 25.268482490272373
ERROR:model_theirs:Full valid score (1) with UCB1: 0.8893360160965795, average step 14.17117117117117
ERROR:model_theirs:Best model (1) at epoch 1, score 0.8893360160965795, average step 14.17117117117117. BEST_MODELS: [4, 1, 0]
ERROR:model_theirs:Full test score (1) with UCB1: 0.8306403351286654, average step 21.297665369649806
ERROR:model_theirs:Full valid score (1) with UCB1: 0.9215291750503019, average step 15.234234234234235
ERROR:model_theirs:Best model (1) at epoch 2, score 0.9215291750503019, average step 15.234234234234235. BEST_MODELS: [4, 2, 0]
ERROR:model_theirs:Full test score (1) with UCB1: 0.7923399162178336, average step 26.262645914396888
ERROR:model_theirs:Full valid score (1) with UCB1: 0.9004024144869215, average step 13.68018018018018
ERROR:model_theirs:Full valid score (1) with UCB1: 0.9336016096579477, average step 13.76126126126126
ERROR:model_theirs:Best model (1) at epoch 4, score 0.9336016096579477, average step 13.76126126126126. BEST_MODELS: [4, 4, 0]
ERROR:model_theirs:Full test score (1) with UCB1: 0.8674446439257929, average step 19.027237354085603
ERROR:model_theirs:Full valid score (2) with UCB1: 0.9275653923541247, average step 15.932432432432432
ERROR:model_theirs:Best model (2) at epoch 0, score 0.9275653923541247, average step 15.932432432432432. BEST_MODELS: [4, 4, 0]
ERROR:model_theirs:Full test score (2) with UCB1: 0.8138839018551766, average step 28.84046692607004
ERROR:model_theirs:Full valid score (2) with UCB1: 0.903420523138833, average step 14.716216216216216
ERROR:model_theirs:Full valid score (2) with UCB1: 0.9356136820925554, average step 14.86936936936937
ERROR:model_theirs:Best model (2) at epoch 2, score 0.9356136820925554, average step 14.86936936936937. BEST_MODELS: [4, 4, 2]
ERROR:model_theirs:Full test score (2) with UCB1: 0.8782166367444644, average step 22.43190661478599
ERROR:model_theirs:Full valid score (2) with UCB1: 0.8913480885311871, average step 12.878378378378379
ERROR:model_theirs:Full valid score (2) with UCB1: 0.8933601609657947, average step 12.81981981981982
```


## 6.8

```log
Validating test games: 100%|████████████████████████████████████████████████████████████████████████████████████| 514/514 [32:35<00:00,  3.80s/it]
Full test score (0): 0.895870736086176 with UCB1, average step 22.01556420233463
Validating test games: 100%|████████████████████████████████████████████████████████████████████████████████████| 514/514 [29:20<00:00,  3.43s/it]
Full test score (1): 0.8764213046080191 with UCB1, average step 19.470817120622566
Validating test games: 100%|████████████████████████████████████████████████████████████████████████████████████| 514/514 [31:21<00:00,  3.66s/it]
Full test score (2): 0.8638539796529024 with UCB1, average step 22.69260700389105
```

## 6.9 valid set

```log

Validating valid games: 100%|███████████████████████████████████████████████████████████████████████████████████| 222/222 [08:43<00:00,  2.36s/it]
Full valid score (0): 0.9406438631790744 with UCB1, average step 15.13063063063063
Validating valid games: 100%|███████████████████████████████████████████████████████████████████████████████████| 222/222 [08:02<00:00,  2.17s/it]
Full valid score (1): 0.9325955734406438 with UCB1, average step 13.878378378378379
Validating valid games: 100%|███████████████████████████████████████████████████████████████████████████████████| 222/222 [08:28<00:00,  2.29s/it]
Full valid score (2): 0.9356136820925554 with UCB1, average step 14.882882882882884
```