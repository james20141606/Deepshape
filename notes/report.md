## 2017.5.15
* Redid Deepfold prediction on the Spitale_2015_invitro and Spitale_2015_invivo dataset using mlp1, conv1 and logreg and window size 100
* Moved the reports/MetricTable to legacy/MetricTable.stride=10
* Compare the performance after increasing the sample size
```bash
cat reports/MetricTable/icSHAPE/*.txt | awk '( index($1, "Spitale_2015") != 0) && ($4 == 100) && (($5 == "mlp1") ||
($5 == "conv1") || ($5 == "logreg"))' > tmp/MetricTable.stride=1.txt
cat legacy/MetricTable.stride=10/icSHAPE/*.txt | awk '( index($1, "Spitale_2015") != 0) && ($4 == 100) && (($5 == "mlp1") ||
($5 == "conv1") || ($5 == "logreg"))' > tmp/MetricTable.stride=10.txt
```
