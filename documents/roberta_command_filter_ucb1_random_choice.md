2025.5.7

RoBERTa + command generate + command filter + `randomized command indexs`

唯一的不同点在于`randomized command indexs`。

* 过去的模型（平均测试分数0.694）中屡次观测到选择致命选项的情况。
* 仔细分析得知模型会赋予同类型选项较高的分数。
* 这个事情本身并没有不合逻辑——这意味着模型知道现在应该做什么。
* 但是这样的分数分布是危险的，因为跟正确选项同类型的选项很可能是致命选项。
* 为了促进模型学习细粒度的区分，我们认为将指令打乱是一个好办法，这样将迫使模型选择具体的选项，而不是指向一个选项群。

```log
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.6810865191146881, average step 15.063063063063064
ERROR:model_ours:Best model (0) at epoch 0, score 0.6810865191146881, average step 15.063063063063064. BEST_MODELS: [0, 0, 0]
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8350100603621731, average step 21.88738738738739
ERROR:model_ours:Best model (0) at epoch 1, score 0.8350100603621731, average step 21.88738738738739. BEST_MODELS: [1, 0, 0]
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8128772635814889, average step 24.87837837837838
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8199195171026157, average step 24.48198198198198
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.886317907444668, average step 23.12162162162162
ERROR:model_ours:Best model (0) at epoch 4, score 0.886317907444668, average step 23.12162162162162. BEST_MODELS: [4, 0, 0]
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9114688128772636, average step 18.0990990990991
ERROR:model_ours:Best model (0) at epoch 5, score 0.9114688128772636, average step 18.0990990990991. BEST_MODELS: [5, 0, 0]
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8400402414486922, average step 19.103603603603602
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8329979879275654, average step 22.283783783783782
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.6891348088531187, average step 17.03153153153153
ERROR:model_ours:Best model (1) at epoch 0, score 0.6891348088531187, average step 17.03153153153153. BEST_MODELS: [5, 0, 0]
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8470824949698189, average step 18.833333333333332
ERROR:model_ours:Best model (1) at epoch 1, score 0.8470824949698189, average step 18.833333333333332. BEST_MODELS: [5, 1, 0]
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8410462776659959, average step 24.324324324324323
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.693158953722334, average step 18.2027027027027
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8249496981891348, average step 24.734234234234233
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8551307847082495, average step 20.774774774774773
ERROR:model_ours:Best model (1) at epoch 5, score 0.8551307847082495, average step 20.774774774774773. BEST_MODELS: [5, 5, 0]
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.829979879275654, average step 23.40990990990991
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8158953722334004, average step 22.81081081081081
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.6871227364185111, average step 23.914414414414413
ERROR:model_ours:Best model (2) at epoch 0, score 0.6871227364185111, average step 23.914414414414413. BEST_MODELS: [5, 5, 0]
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.7334004024144869, average step 26.18018018018018
ERROR:model_ours:Best model (2) at epoch 1, score 0.7334004024144869, average step 26.18018018018018. BEST_MODELS: [5, 5, 1]
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8712273641851107, average step 19.59009009009009
ERROR:model_ours:Best model (2) at epoch 2, score 0.8712273641851107, average step 19.59009009009009. BEST_MODELS: [5, 5, 2]
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8682092555331992, average step 16.94144144144144
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8611670020120724, average step 23.783783783783782
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9245472837022133, average step 17.68918918918919
ERROR:model_ours:Best model (2) at epoch 5, score 0.9245472837022133, average step 17.68918918918919. BEST_MODELS: [5, 5, 5]
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9195171026156942, average step 18.675675675675677
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9305835010060363, average step 18.175675675675677
ERROR:model_ours:Best model (2) at epoch 7, score 0.9305835010060363, average step 18.175675675675677. BEST_MODELS: [5, 5, 7]
ERROR:model_ours:vvvvv
Testing trained models
ERROR:model_ours:Best models: [5, 5, 7]
ERROR:model_ours:Full valid score (0): 0.8822937625754527 with UCB1, average step 14.923423423423424
ERROR:model_ours:Full test score (0): 0.7692998204667864 with UCB1, average step 23.616731517509727
ERROR:model_ours:Full valid score (1): 0.8591549295774648 with UCB1, average step 14.923423423423424
ERROR:model_ours:Full test score (1): 0.7812687013764213 with UCB1, average step 21.883268482490273
ERROR:model_ours:Full valid score (2): 0.9164989939637826 with UCB1, average step 15.35135135135135
ERROR:model_ours:Full test score (2): 0.8216636744464393 with UCB1, average step 22.054474708171206
 
```

Without UCB1 test.

```log
ERROR:model_ours:vvvvv
Testing trained models w/o UCB1
ERROR:model_ours:Best models: [5, 5, 7]
ERROR:model_ours:Full valid score (0): 0.8722334004024145 w/o UCB1, average step 18.68018018018018
ERROR:model_ours:Full test score (0): 0.7797725912627169 w/o UCB1, average step 28.373540856031127
ERROR:model_ours:Full valid score (1): 0.8531187122736419 w/o UCB1, average step 19.864864864864863
ERROR:model_ours:Full test score (1): 0.7618192698982645 w/o UCB1, average step 31.058365758754864
ERROR:model_ours:Full valid score (2): 0.937625754527163 w/o UCB1, average step 18.90990990990991
ERROR:model_ours:Full test score (2): 0.8210652304009575 w/o UCB1, average step 28.38715953307393
```