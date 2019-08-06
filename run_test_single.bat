set img_name=DSC00132_tiny
set test_img=dataset/test/DSC00132_tiny.jpg
set model_name=pspnet-densenet-s1s2
set epoch=130
set model=snapshot/%model_name%/%epoch%.pkl
set save_dir=results\%model_name%\%epoch%
mkdir %save_dir%
mkdir %save_dir%\temp
if not exist %save_dir%/%img_name%_pred.png (
    echo file exists:%save_dir%\%img_name%_pred.png
)
python test.py --img_path %test_img% ^
                --out_path %save_dir%/%img_name%_pred.png ^
                --vis_out_path %save_dir%/vis_%img_name%_pred.png ^
                --gpu 0 ^
                --batch_size 8 ^
                --stride 64 ^
                --model_path %model% ^
                --input_size 256 ^
                --crop_scales 192 224 256 288 320 ^
                --tempdir %save_dir%/temp