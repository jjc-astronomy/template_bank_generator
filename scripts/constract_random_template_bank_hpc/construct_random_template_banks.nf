nextflow.enable.dsl=2
project_dir = projectDir

process run_part {
    label 'emcee'
    //container "${params.apptainer_images.pulsarx}"
    errorStrategy 'ignore'
    publishDir "$projectDir", mode: 'copy'

    input:
    val file_index
    val ntemplates_part


    script:
    def seed = file_index + 42
    def part_string = String.format('%03d', file_index)
    """
    python $project_dir/construct_random_template_bank_emcee_part.py -t $params.tobs -p $params.porb_min -P $params.porb_max -c $params.max_companion_mass -d $params.min_pulsar_mass -s $params.spin_period -f $params.inclination_angle_fraction -b $params.coverage -m $params.mismatch -n $task.cpus -z $ntemplates_part --seed $seed --burnin $params.burnin --thin $params.thin --nwalkers $params.nwalkers -o ${params.output_prefix}_part$part_string
    """
}


workflow{
    chan_index = Channel.from(0..(params.nfiles - 1))
    ntemplates_first = Math.round(params.templates / params.nfiles)
    ntemplates_last = params.templates - (params.nfiles - 1) * ntemplates_first
    ntemplates_list = []
    for (i in 0..(params.nfiles - 2)) {
        ntemplates_list.add(ntemplates_first)
    }
    ntemplates_list.add(ntemplates_last)
    chan_ntemplates = Channel.from(ntemplates_list)
    run_part(chan_index, chan_ntemplates)
}