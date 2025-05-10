## 2025.5.10 追加使用ucb1的实验

valid: [0.9215291750503019, 0.903420523138833, 0.8983903420523138] = 0.908
test: [0.824655894673848, 0.7848593656493118, 0.8303411131059246] = 0.813

```log
ERROR:model_ours:vvvvv
Testing trained models with UCB1
ERROR:model_ours:Best models: [3, 2, 3]
ERROR:model_ours:Full valid score (0): 0.9215291750503019 with UCB1, average step 16.027027027027028
ERROR:model_ours:Full test score (0): 0.824655894673848 with UCB1, average step 24.93579766536965
ERROR:model_ours:Full valid score (1): 0.903420523138833 with UCB1, average step 15.216216216216216
ERROR:model_ours:Full test score (1): 0.7848593656493118 with UCB1, average step 21.30739299610895
ERROR:model_ours:Full valid score (2): 0.8983903420523138 with UCB1, average step 15.17117117117117
ERROR:model_ours:Full test score (2): 0.8303411131059246 with UCB1, average step 24.34046692607004
```

## 2025.5.10 改变训练率 + 训练回合数

不使用ucb1, 只训练5回合： 
BEST: [3, 2, 3]
valid: [0.9396378269617707, 0.9195171026156942, 0.9064386317907445] = 0.922
test: [0.8204667863554758, 0.7743865948533812, 0.817474566128067] = 0.804

不使用ucb1, 训练8回合：
BEST: [7, 2, 7]
valid: [0.9466800804828974, 0.9195171026156942, 0.9527162977867203] = 0.940
test: [0.8402154398563735, 0.7743865948533812, 0.8473967684021544] = 0.820

平均提高了1.7%左右，为了节省时间，选择放弃掉这一点性能提升。


```log
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.6187122736418511, average step 13.801801801801801
ERROR:model_ours:Best model (0) at epoch 0, score 0.6187122736418511, average step 13.801801801801801. BEST_MODELS: [0, 0, 0]
ERROR:model_ours:Full test score (0) w/o UCB1: 0.5517654099341711, average step 22.01750972762646
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9054325955734407, average step 18.03153153153153
ERROR:model_ours:Best model (0) at epoch 1, score 0.9054325955734407, average step 18.03153153153153. BEST_MODELS: [1, 0, 0]
ERROR:model_ours:Full test score (0) w/o UCB1: 0.7516457211250748, average step 28.77821011673152
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.8370221327967807, average step 19.207207207207208
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9396378269617707, average step 18.95045045045045
ERROR:model_ours:Best model (0) at epoch 3, score 0.9396378269617707, average step 18.95045045045045. BEST_MODELS: [3, 0, 0]
ERROR:model_ours:Full test score (0) w/o UCB1: 0.8204667863554758, average step 30.591439688715955
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.886317907444668, average step 19.51801801801802
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9235412474849095, average step 20.50900900900901
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9164989939637826, average step 20.315315315315317
ERROR:model_ours:Full valid score (0) w/o UCB1: 0.9466800804828974, average step 19.513513513513512
ERROR:model_ours:Best model (0) at epoch 7, score 0.9466800804828974, average step 19.513513513513512. BEST_MODELS: [7, 0, 0]
ERROR:model_ours:Full test score (0) w/o UCB1: 0.8402154398563735, average step 28.929961089494164
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.6207243460764588, average step 15.432432432432432
ERROR:model_ours:Best model (1) at epoch 0, score 0.6207243460764588, average step 15.432432432432432. BEST_MODELS: [7, 0, 0]
ERROR:model_ours:Full test score (1) w/o UCB1: 0.5263315380011969, average step 22.027237354085603
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8782696177062375, average step 17.63963963963964
ERROR:model_ours:Best model (1) at epoch 1, score 0.8782696177062375, average step 17.63963963963964. BEST_MODELS: [7, 1, 0]
ERROR:model_ours:Full test score (1) w/o UCB1: 0.7657091561938959, average step 30.336575875486382
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.9195171026156942, average step 17.463963963963963
ERROR:model_ours:Best model (1) at epoch 2, score 0.9195171026156942, average step 17.463963963963963. BEST_MODELS: [7, 2, 0]
ERROR:model_ours:Full test score (1) w/o UCB1: 0.7743865948533812, average step 26.056420233463037
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8843058350100603, average step 17.75225225225225
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.8903420523138833, average step 21.693693693693692
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.9154929577464789, average step 19.60810810810811
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.9175050301810865, average step 18.036036036036037
ERROR:model_ours:Full valid score (1) w/o UCB1: 0.9024144869215291, average step 20.743243243243242
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.5935613682092555, average step 18.436936936936938
ERROR:model_ours:Best model (2) at epoch 0, score 0.5935613682092555, average step 18.436936936936938. BEST_MODELS: [7, 2, 0]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.5248354278874925, average step 28.94941634241245
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8682092555331992, average step 16.7972972972973
ERROR:model_ours:Best model (2) at epoch 1, score 0.8682092555331992, average step 16.7972972972973. BEST_MODELS: [7, 2, 1]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.7330939557151407, average step 26.957198443579767
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8792756539235412, average step 19.5990990990991
ERROR:model_ours:Best model (2) at epoch 2, score 0.8792756539235412, average step 19.5990990990991. BEST_MODELS: [7, 2, 2]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.7555356074207061, average step 31.268482490272373
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9064386317907445, average step 20.31981981981982
ERROR:model_ours:Best model (2) at epoch 3, score 0.9064386317907445, average step 20.31981981981982. BEST_MODELS: [7, 2, 3]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.817474566128067, average step 31.714007782101167
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.8883299798792756, average step 23.216216216216218
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9014084507042254, average step 19.60810810810811
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9154929577464789, average step 20.36936936936937
ERROR:model_ours:Best model (2) at epoch 6, score 0.9154929577464789, average step 20.36936936936937. BEST_MODELS: [7, 2, 6]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.8426092160383004, average step 30.509727626459146
ERROR:model_ours:Full valid score (2) w/o UCB1: 0.9527162977867203, average step 19.463963963963963
ERROR:model_ours:Best model (2) at epoch 7, score 0.9527162977867203, average step 19.463963963963963. BEST_MODELS: [7, 2, 7]
ERROR:model_ours:Full test score (2) w/o UCB1: 0.8473967684021544, average step 28.573929961089494
ERROR:model_ours:vvvvv
Testing trained models w/o UCB1
ERROR:model_ours:Best models: [7, 2, 7]
ERROR:model_ours:Full valid score (0): 0.9446680080482898 w/o UCB1, average step 19.47747747747748
ERROR:model_ours:Full test score (0): 0.8464991023339318 w/o UCB1, average step 27.852140077821012
ERROR:model_ours:Full valid score (1): 0.8822937625754527 w/o UCB1, average step 17.81081081081081
ERROR:model_ours:Full test score (1): 0.7878515858767206 w/o UCB1, average step 26.377431906614785
ERROR:model_ours:Full valid score (2): 0.9426559356136821 w/o UCB1, average step 21.97747747747748
ERROR:model_ours:Full test score (2): 0.8566726511071214 w/o UCB1, average step 29.523346303501945
```