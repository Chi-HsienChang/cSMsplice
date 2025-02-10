# cSMsplice

## light TAIR10.fa 
## without --prelearned_sres arabidopsis 
## with --learn_sres --print_local_scores --learning_seed
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10_light.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10_light.fa -m ./maxEnt_models/arabidopsis/ --print_predictions --print_local_scores --learn_sres --learning_seed real-decoy

-------------------
-------------------
## original TAIR10.fa
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10.fa -m ./maxEnt_models/arabidopsis/ --prelearned_sres arabidopsis --print_predictions

## light TAIR10.fa with --prelearned_sres arabidopsis
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10_light.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10_light.fa -m ./maxEnt_models/arabidopsis/ --prelearned_sres arabidopsis --print_predictions

## light TAIR10.fa without --prelearned_sres arabidopsis
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10_light.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10_light.fa -m ./maxEnt_models/arabidopsis/ --print_predictions

## light TAIR10.fa without --prelearned_sres arabidopsis with --learn_sres
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10_light.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10_light.fa -m ./maxEnt_models/arabidopsis/ --print_predictions
--learn_sres

## light TAIR10.fa 
## without --prelearned_sres arabidopsis 
## with --learn_sres --print_local_scores --learning_seed
time python3 runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10_light.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g ./dataset/TAIR10_light.fa -m ./maxEnt_models/arabidopsis/ --print_predictions --print_local_scores --learn_sres --learning_seed real-decoy






