#!/bin/bash

home="/data6/tan77/opt/Interformer"
po=MP
DOCK_FOLDER=energy_output

for rec in `cat LST_PRO`; do
    rec1=`awk -F_ '{print $2}' <<< $rec`
	if [[ ! -d ./$rec ]]; then
		mkdir $rec
	fi
    for lig in `ls for_SI_MIN/$rec/$po/*.mol2 | grep -v sybyl`; do
        
        lig_name=$(basename $lig .mol2)
        if [[ -f ./output/$rec/$lig_name/input.round0_ensemble.csv ]]; then
            echo "$rec $lig_name done"
            continue
        fi
        
        if [[ ! -d ./$rec/$lig_name ]]; then
            mkdir $rec/$lig_name
            mkdir $rec/$lig_name/ligand
            mkdir $rec/$lig_name/pocket
            mkdir $rec/$lig_name/uff
            mkdir $rec/$lig_name/raw
            mkdir $rec/$lig_name/tmp_lig
        fi
        # if [ ! -f $rec/$lig_name/log ]; then
        #     touch $rec/$lig_name/log
        # fi

        ref=`head -n 1 ./for_SI/$rec/ref`
        echo "---------- Step 1: prepare reference for $rec $lig_name ----------"
        antechamber \
        -fi mol2 \
        -fo sdf \
        -i ./for_SI_MIN/$rec/$po/$ref.mol2 \
        -o ./$rec/$lig_name/ligand/$rec1\_docked.sdf \
        -at sybyl #>> $rec/$lig_name/log

        cp ./for_SI_MIN/$rec/$po/$lig_name.pdb ./$rec/$lig_name/pocket/$rec1\_pocket.pdb
        sed -e "s/%NAME%/$rec1/g" -e "s/%RECNAME%/$rec/g" input_template.csv > ./$rec/$lig_name/input.csv

        echo "---------- Step 2: prepare ligand for $rec $lig_name ----------"
        antechamber \
        -fi mol2 \
        -fo sdf \
        -i for_SI_MIN/$rec/$po/$lig_name.mol2 \
        -o $rec/$lig_name/tmp_lig/$rec1.sdf \
        -at sybyl \
        -rn $lig_name #>> $rec/$lig_name/log

        echo "---------- Step 3: generate 3D conformers for $rec $lig_name ----------"
        python $home/tools/rdkit_ETKDG_3d_gen.py $rec/$lig_name/tmp_lig/ $rec/$lig_name/uff #>> $rec/$lig_name/log

        echo "---------- Step 4: run Interformer for $rec $lig_name ----------"
        PYTHONPATH=$home/interformer/ python $home/inference.py \
        -test_csv ./$rec/$lig_name/input.csv \
        -work_path ./$rec/$lig_name/ \
        -ensemble $home/checkpoints/v0.2_energy_model \
        -batch_size 1 \
        -posfix *val_loss* \
        -energy_output_folder ./$rec/$lig_name/$DOCK_FOLDER \
        -reload \
        -debug #>> $rec/$lig_name/log

        echo "---------- Step 5: reconstruct ligand for $rec $lig_name ----------"
        OMP_NUM_THREADS="64,64" python $home/docking/reconstruct_ligands.py \
        -y \
        --cwd ./$rec/$lig_name/$DOCK_FOLDER \
        -y \
        --find_all find #>> $rec/$lig_name/log

        echo "---------- Step 6: merge summary and input for $rec $lig_name ----------"
        python $home/docking/reconstruct_ligands.py \
        --cwd ./$rec/$lig_name/$DOCK_FOLDER \
        --find_all stat #>> $rec/$lig_name/log

        echo "---------- Step 7: prepare input for ensemble model for $rec $lig_name ----------"
        python $home/docking/merge_summary_input.py \
        ./$rec/$lig_name/$DOCK_FOLDER/ligand_reconstructing/stat_concated.csv ./$rec/$lig_name/input.csv #>> $rec/$lig_name/log

        mkdir -p ./$rec/$lig_name/infer && cp -r ./$rec/$lig_name/$DOCK_FOLDER/ligand_reconstructing/*.sdf ./$rec/$lig_name/infer

        if [ -d "./$rec/$lig_name/tmp_beta" ]; then
            rm -r ./$rec/$lig_name/tmp_beta
        fi

        echo "---------- Step 8: run ensemble model for $rec $lig_name ----------"
        PYTHONPATH=$home/interformer/ python $home/inference.py \
        -test_csv ./$rec/$lig_name/input.round0.csv \
        -work_path ./$rec/$lig_name/ \
        -ligand_folder infer/ \
        -ensemble $home/checkpoints/v0.2_affinity_model/model* \
        -gpus 1 \
        -batch_size 20 \
        -posfix *val_loss* \
        --pose_sel True #>> $rec/$lig_name/log
        
	if [ ! -d ./output/$rec/$lig_name ]; then
		mkdir -p ./output/$rec/$lig_name
	fi
	mv result/* ./output/$rec/$lig_name
    done
done
