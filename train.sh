#nohup python3 train_BDNet_real_static.py > train.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian5.yml > train5.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian15.yml > train15.log 2>&1 &
nohup python3 train_BDNet_real_static.py --opt_file=options/train/train_BDNet_gaussian25.yml > train25.log 2>&1 &
