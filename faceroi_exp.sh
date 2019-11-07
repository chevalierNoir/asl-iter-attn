#! /bin/bash

. ./path.sh
parts=(train dev test)
stage=1
config=./conf.ini

iters=(0 1 2)
scales=(0.721 0.721 1.0)
epochs=(35 35 35)
hw=(224 224)
map_size=13

if [ $stage -eq 1 ];then
    echo "training"
    for iter in ${iters[@]};do
        sid=0
        eid=-1

        echo $iter

        # resize prior
        python -B make_map.py --img $iter_dir/prior_$iter --mat $iter_dir/priormat_$iter --csv $whole_csv --size $map_size --start $sid --end $eid || exit 1;

        # train
        model_dir=$iter_dir/model_$iter
        python -B train.py --img $iter_dir/rgb_$iter --prior $iter_dir/priormat_$iter --csv $csv_dir --output $model_dir --pretrain $prt_conv --epoch ${epochs[$iter]} --conf $config || exit 1;

        # evaluation (attn map)
        beta_dir=$iter_dir/beta_$iter
        python -B evaluate.py --img $iter_dir/rgb_$iter --prior $iter_dir/priormat_$iter --csv $csv_dir --output $beta_dir --model $model_dir/best.pth --task beta --conf $config || exit 1;

        if [ $iter -lt ${iters[-1]} ];then
            echo $iter
            # cropping
            next_iter=$(($iter+1))
            scale=${scales[$iter]}
            lambda_dirs=`for i in $(seq 0 $next_iter);do printf "%s%s " $iter_dir/lambda_ $i;done`
            python -B aug_res.py --beta $iter_dir/beta_$iter --lambda_ $iter_dir/lambda_$next_iter --csv $whole_csv --scale $scale --save_lambda --avg 1 --start $sid --end $eid --hw ${hw[@]} || exit 1;

            python -B aug_res.py --beta $iter_dir/beta_$iter --lambda_ ${lambda_dirs[@]} --input $rgb_dir --output $iter_dir/rgb_$next_iter --csv $whole_csv --scale $scale --save_img --avg 1 --start $sid --end $eid --hw ${hw[@]} || exit 1;

            python -B aug_res.py --beta $iter_dir/beta_$iter --lambda_ ${lambda_dirs[@]} --input $prior_dir --output $iter_dir/prior_$next_iter --scale $scale --csv $whole_csv --save_img --avg 1 --start $sid --end $eid --hw ${hw[@]} || exit 1;
        fi
    done

elif [ $stage -eq 2 ];then
     echo "evaluating"
     iter=${iters[-1]}
     partition=test
     prob_dir=$iter_dir/prob_$iter
     model_dir=$iter_dir/model_$iter
     python -B evaluate.py --img $iter_dir/rgb_$iter --prior $iter_dir/priormat_$iter --csv $csv_dir --output $prob_dir --model $model_dir/best.pth --partition $partition --conf $config --task prob || exit 1;

     echo "Greedy decoding"
     python -B decode.py --csv $csv_dir/$partition.csv --pred $prob_dir --decode_type greedy --conf $config || exit 1;

     echo "Beam Search with LM: (1). train LM (2). beam search"
     python -B lm/train_lm.py --conf $config --output $lm_dir --train $csv_dir/train.csv --dev $csv_dir/dev.csv || exit 1;

     python -B decode.py --csv $csv_dir/$partition.csv --pred $prob_dir --decode_type beam --beam_size 5 --lm_beta 0.1 --lm_pth $lm_dir/best.pth --ins_gamma 0.3 --conf $config || exit 1;
fi
