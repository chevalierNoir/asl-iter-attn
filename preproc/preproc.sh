#! /bin/bash

. ../path.sh

# start sequence id
start=0
# end sequence id (-1: until the end)
end=2

echo "process csv file"
python -B proc_csv.py --iraw $pd_csv --odir $csv_dir --ofn $whole_csv || exit 1;

echo "Running optical flow"
python -B batch_dense_opt.py --start $start --end $end --rgb $rgb_dir --opt $opt_dir --csv $whole_csv || exit 1;

echo "Getting prior"
python -B make_prior.py --OPT $opt_dir --csv $whole_csv --PRIOR $prior_dir --start $start --end $end || exit 1;

# task: face-roi or whole
task=face-roi

if [ $task == 'whole' ];then
    echo "Resizing frame"
    init_rgb_dir=$iter_dir/rgb_0
    init_prior_dir=$iter_dir/prior_0
    init_lambda_dir=$iter_dir/lambda_0
    python -B resize_frame.py --input_dir $rgb_dir --output_dir $init_rgb_dir --lambda_dir $init_lambda_dir --csv $whole_csv --df $pd_csv --start $start --end $end || exit 1;
    python -B resize_frame.py --input_dir $prior_dir --output_dir $init_prior_dir --csv $whole_csv --df $pd_csv --start $start --end $end || exit 1;

elif [ $task == 'face-roi' ];then
    echo "Face detection"
    origin_bbox_dir=$data_dir/face_bbox
    # python -B detect_face.py --rgb $rgb_dir --bbox $origin_bbox_dir --csv $whole_csv --start $start --end $end || exit 1;

    init_rgb_dir=$iter_dir/rgb_0
    init_prior_dir=$iter_dir/prior_0
    init_lambda_dir=$iter_dir/lambda_0

    echo "Processing face bbox"

    signer_bbox_dir=$data_dir/signer_bbox
    log=$data_dir/face_bbox/nonface
    python -B proc_face.py --task face --input_dir $origin_bbox_dir --output_dir $signer_bbox_dir --opt_dir $prior_dir --log_dir $log.$start --csv $whole_csv --start $start --end $end || exit 1;
    python -B proc_zero.py --rgb $rgb_dir --bbox $signer_bbox_dir --csv $whole_csv --zero $log.$start || exit 1;

    echo "Crop Face ROI"
    python -B proc_face.py --task crop --input_dir $rgb_dir --output_dir $init_rgb_dir --lambda_dir $init_lambda_dir --face_dir $signer_bbox_dir --csv $whole_csv --start $start --end $end || exit 1;
    python -B proc_face.py --task crop --input_dir $prior_dir --output_dir $init_prior_dir --lambda_dir $init_lambda_dir\_tmp --face_dir $signer_bbox_dir --csv $whole_csv --start $start --end $end || exit 1;

else
    echo "task: whole | face-roi"
fi
