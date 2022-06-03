# Softmax/UCB Exploring MAB Problem
Playing a 10-armed Bandit and making a lot of money :)

## Softmax
```
➜   python softmax_bandits.py

Total reward: 99.74%

   Predicted      Real
0  -1.026085 -1.028988
1   0.279060  0.289400
2  -0.462657 -0.467154
3   0.334152  0.303548
4  -0.423949 -0.351552
5   0.311439  0.280261
6  -0.279752 -0.221678
7   1.394923  1.393370
8  -0.761078 -0.743508
9  -0.073140 -0.149143

           Softmax
Arm
0    0.0000000000%
1    0.0000142395%
2    0.0000000086%
3    0.0000247044%
4    0.0000000126%
5    0.0000196846%
6    0.0000000533%
7    0.9999408760%
8    0.0000000004%
9    0.0000004205%
```

## UCB
```
➜   python confidence_bandits.py

Total reward: 100.09%

       Real  Predicted  Predicted UCB
0 -2.688990  -4.380469      -4.041162
1  0.017447   0.818183       1.157490
2  0.299733   0.123528       0.462835
3 -0.478235  -2.155562      -1.816255
4  0.173667  -1.281772      -0.942465
5  1.191158   0.999569       1.239495
6  1.975792   1.977806       1.978883
7  0.868813   0.332492       0.671799
8 -0.345694   0.945941       1.285248
9  0.845732   0.820907       1.160214
```
