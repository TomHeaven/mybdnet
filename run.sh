#nohup python3 train_BDNet_real_static.py > run.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian5.yml > run5.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian15.yml > run15.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian25.yml > run25.log 2>&1 &
