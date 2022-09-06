base="/lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs"

mylist=(0 10 20 30 40)

for i in ${mylist[@]};
    do
        cp $base/templates/mashnet_s4k.slurm $base/mashnet_$i.slurm

        sed -i "s/CVNUM/$i/" $base/mashnet_$i.slurm

        sbatch $base/mashnet_$i.slurm

        rm $base/mashnet_$i.slurm

    done