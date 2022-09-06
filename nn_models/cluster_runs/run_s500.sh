base="/lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs"

for i in {0..49}
    do
        cp $base/templates/mashnet_s500.slurm $base/mashnet_$i.slurm

        sed -i "s/CVNUM/$i/" $base/mashnet_$i.slurm

        sbatch $base/mashnet_$i.slurm

        rm $base/mashnet_$i.slurm

    done