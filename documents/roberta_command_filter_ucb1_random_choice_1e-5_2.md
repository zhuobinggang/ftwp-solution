基本和`roberta_command_filter_ucb1_random_choice_1e-5`一样，只是不确定ucb1的实现是否有bug（主要是从state获取指令的时候会打乱顺序，导致ucb1根本搞错了下标，但是结果似乎差不多）

w/o ucb1

* valid: [0.9074446680080482, 0.8802816901408451, 0.9275653923541247]
* test: [0.779473369239976, 0.7830640335128666, 0.8273488928785159]

with ucb1

* valid: [0.8853118712273642, 0.8661971830985915, 0.9265593561368209]
* test: [0.7875523638539796, 0.7923399162178336, 0.824655894673848]

BEST: [3, 1, 4]

```log
Full valid score (0) w/o UCB1: 0.6177062374245473, average step 13.085585585585585
Full test score (0) w/o UCB1: 0.5317175344105326, average step 21.361867704280154
Full valid score (0) w/o UCB1: 0.6488933601609658, average step 16.576576576576578
Full test score (0) w/o UCB1: 0.5394973070017953, average step 25.799610894941633
Full valid score (0) w/o UCB1: 0.8792756539235412, average step 18.7027027027027
Full test score (0) w/o UCB1: 0.7444643925792939, average step 28.59727626459144
Full valid score (0) w/o UCB1: 0.9074446680080482, average step 19.504504504504503
Full test score (0) w/o UCB1: 0.779473369239976, average step 29.801556420233464
Full valid score (0) w/o UCB1: 0.8822937625754527, average step 19.554054054054053
Full valid score (1) w/o UCB1: 0.6267605633802817, average step 19.274774774774773
Full test score (1) w/o UCB1: 0.5631358467983244, average step 26.14980544747082
Full valid score (1) w/o UCB1: 0.8802816901408451, average step 19.882882882882882
Full test score (1) w/o UCB1: 0.7830640335128666, average step 32.7431906614786
Full valid score (1) w/o UCB1: 0.817907444668008, average step 25.00900900900901
Full valid score (1) w/o UCB1: 0.8752515090543259, average step 20.43243243243243
Full valid score (1) w/o UCB1: 0.8309859154929577, average step 21.972972972972972
Full valid score (2) w/o UCB1: 0.6267605633802817, average step 18.216216216216218
Full test score (2) w/o UCB1: 0.5526630760023937, average step 24.782101167315176
Full valid score (2) w/o UCB1: 0.8561368209255533, average step 21.42792792792793
Full test score (2) w/o UCB1: 0.7369838420107719, average step 28.56809338521401
Full valid score (2) w/o UCB1: 0.8983903420523138, average step 20.28828828828829
Full test score (2) w/o UCB1: 0.7761819269898265, average step 33.89299610894942
Full valid score (2) w/o UCB1: 0.9175050301810865, average step 20.193693693693692
Full test score (2) w/o UCB1: 0.8345302214242968, average step 29.394941634241246
Full valid score (2) w/o UCB1: 0.9275653923541247, average step 21.405405405405407
Full test score (2) w/o UCB1: 0.8273488928785159, average step 29.99805447470817

Full valid score (0): 0.8853118712273642 with UCB1, average step 16.86036036036036
Full test score (0): 0.7875523638539796 with UCB1, average step 25.556420233463037
Full valid score (1): 0.8661971830985915 with UCB1, average step 15.292792792792794
Full test score (1): 0.7923399162178336 with UCB1, average step 24.328793774319067
Full valid score (2): 0.9265593561368209 with UCB1, average step 17.03153153153153
Full test score (2): 0.824655894673848 with UCB1, average step 24.254863813229573
```