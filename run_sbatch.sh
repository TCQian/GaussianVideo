# gaussian_image
dataset=("Honeybee" "Beauty" "Jockey")
testing_gaussians_1frames=(750 1500 2250 3000 3750 7500 15000 22500 30000 37500)
for i in "${!dataset[@]}"; do
    for j in "${!testing_gaussians[@]}"; do
        sbatch run_gaussianimage.sh \
            --data_name "${dataset[$i]}" \
            --num_points "${testing_gaussians[$j]}" \
    done
done

# gaussian_video
num_frames=(1 5 10 20)
testing_gaussians_5frames=(750 1500 2250 3000 3750 7500 15000 22500 30000 37500)
testing_gaussians_10frames=(5000 10000 15000 20000 25000 50000 100000 150000 200000 250000)
testing_gaussians_20frames=(10000 20000 30000 40000 50000 100000 200000 300000 400000 500000)
for i in "${!dataset[@]}"; do    
    for frame_idx in "${!num_frames[@]}"; do
        frame_count="${num_frames[$frame_idx]}"

        # Select the corresponding Gaussian list based on the frame count
        if [ "$frame_count" -eq 1 ]; then
            testing_gaussians=("${testing_gaussians_1frames[@]}")
        elif [ "$frame_count" -eq 5 ]; then
            testing_gaussians=("${testing_gaussians_5frames[@]}")
        elif [ "$frame_count" -eq 10 ]; then
            testing_gaussians=("${testing_gaussians_10frames[@]}")
        elif [ "$frame_count" -eq 20 ]; then
            testing_gaussians=("${testing_gaussians_20frames[@]}")
        else
            # Skip unsupported frame counts
            continue
        fi

        # Submit jobs for the current frame count and Gaussian list
        for gaussian in "${testing_gaussians[@]}"; do
            sbatch run_gaussianvideo.sh \
                --data_name "${dataset[$i]}" \
                --num_points "$gaussian" \
                --start_frame 0 \
                --num_frames "$frame_count"
        done
    done
done