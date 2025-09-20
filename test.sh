----------------------------------------------------------------------------------------------------------------------------------------------
# CVC-300 - EndoScene
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/CVC-300" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
# CVC-ClinicDB
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/CVC-ClinicDB" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume ""
# CVC-ColonDB
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/CVC-ColonDB" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume ""
# ETIS-LaribPolypDB
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/ETIS-LaribPolypDB" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume ""
# Kvasir
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/Kvasir" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
-----------------------------------------------------------------------------------------------------------------------------------------------
# COVID19-1
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/COVID19-1" --label_size 512 --image_size 512 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
# COVID19-2
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/COVID19-2" --label_size 512 --image_size 512 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
-----------------------------------------------------------------------------------------------------------------------------------------------
# DSB-2018
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/DSB-2018" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
# MonuSeg-2018
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/MonuSeg-2018" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
-----------------------------------------------------------------------------------------------------------------------------------------------
# BUSI
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/BUSI" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
# STU
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/STU" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "" 
-----------------------------------------------------------------------------------------------------------------------------------------------
