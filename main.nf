nextflow.enable.dsl=2

process split_template_bank{
    label 'split'
    container "${params.apptainer_images.template_bank_generator}"

    input:
    tuple path(template_bank),
    val(nsplits)

    output:
    path "bank_split_*.txt"

    script:
    """
    #!/bin/bash

    #Split the template bank into smaller files

    python ${params.scripts.split_bank} --input_file ${template_bank} --n_splits ${nsplits} --output_dir \$(pwd)

    """

}

process self_check{
    label 'self_check'
    container "${params.apptainer_images.template_bank_generator}"

    input:
    path(template_bank)

    output:
    path("*_self_checked.txt")

    script:
    """
    #!/bin/bash

    #Run self check on the template bank

    output_filename=\$(basename ${template_bank})
    output_filename=\${output_filename%.txt}_self_checked.txt
    python ${params.scripts.self_check} --input_file ${template_bank} --output_file \${output_filename} \
    --tobs ${params.template_bank.tobs} --freq ${params.template_bank.freq} \
    --mismatch_orbital ${params.template_bank.mismatch_orbital} --mismatch_spin_freq ${params.template_bank.mismatch_spin_freq} \
    --nmc ${params.template_bank.montecarlo_integration_iterations} --f_search_range ${params.template_bank.f_search_range} \
    --coarse_threshold ${params.template_bank.coarse_threshold_template} --coarse_step ${params.template_bank.coarse_step_freq} \
    --fine_step ${params.template_bank.fine_step_freq} --log_interval ${params.template_bank.log_interval} --verbose

    """

}

process cross_check {
    label 'cross_check'
    container "${params.apptainer_images.template_bank_generator}"

    input:
    tuple path(template_bank1),
    path(template_bank2),
    val(round_number)

    output:
    path("*_cross_checked.txt")

    script:
    """
    #!/bin/bash

    #Run cross check on the template bank

    output_filename1=\$(basename ${template_bank1})
    filename1_id="\${output_filename1#bank_split_}"
    filename1_id="\${filename1_id%.txt}"
    output_filename2=\$(basename ${template_bank2})
    filename2_id="\${output_filename2#bank_split_}"
    filename2_id="\${filename2_id%.txt}"
    output_filename="round_${round_number}_\${filename1_id}_\${filename2_id}_cross_checked.txt"
   
    python ${params.scripts.pair_check} --bank_file ${template_bank1} --suggested_file ${template_bank2} --output_file \${output_filename} \
    --tobs ${params.template_bank.tobs} --freq ${params.template_bank.freq} \
    --mismatch_orbital ${params.template_bank.mismatch_orbital} --mismatch_spin_freq ${params.template_bank.mismatch_spin_freq} \
    --nmc ${params.template_bank.montecarlo_integration_iterations} --f_search_range ${params.template_bank.f_search_range} \
    --coarse_threshold ${params.template_bank.coarse_threshold_template} --coarse_step ${params.template_bank.coarse_step_freq} \
    --fine_step ${params.template_bank.fine_step_freq} --log_interval ${params.template_bank.log_interval} --nproc ${task.cpus} --verbose

    """
}

template_bank = Channel.fromPath(params.template_bank.random_template_bank)


    

workflow{

    split_input = template_bank.map { bank_file ->
        // Split the template bank into smaller files
        return tuple(bank_file, params.template_bank.nsplits)
    }
    
    // Split the template bank into smaller files
    splitted_bank = split_template_bank(split_input)
    splitted_bank_flattened = splitted_bank.flatten()

    // Run self check on each split template bank file
    checked_bank = self_check(splitted_bank_flattened)

}
