-------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "dataset/Polyp_Train" --work_dir "./work_dir/train_Polyp" --image_size 352 --label_size 352 --num_epochs 300 --batch_size 8 --num_cls 2 --eval_step 3 --lr 5e-4 --use_amp --detailed_metrics > Polyp.log 2>&1
-------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "dataset/DSB-2018" --work_dir "./work_dir/train_DSB-2018_" --image_size 352 --label_size 352 --num_epochs 300 --batch_size 8 --num_cls 2 --eval_step 3 --lr 5e-4 --use_amp --detailed_metrics > DSB-2018.log 2>&1
-------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "dataset/BUSI" --work_dir "./work_dir/train_BUSI" --image_size 352 --label_size 352 --num_epochs 300 --batch_size 8 --num_cls 2 --eval_step 3 --lr 5e-4 --use_amp --detailed_metrics > BUSI.log 2>&1
-------------------------------------------------------------------------------------------------------------------------------------------------

