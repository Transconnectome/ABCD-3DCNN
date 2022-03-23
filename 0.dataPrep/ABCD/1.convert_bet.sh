#!bin/bash
  
for i in `ls /scratch/bigdata/ABCD/freesurfer/smri/freesurfer_smri/*.T1.mgz`
do
#73 from len('/scratch/bigdata/ABCD/freesurfer/smri/freesurfer_smri/sub-NDARINVZZZP87KR') in py
        mri_convert $i "${i:0:73}.T1.nii"
        bet "${i:0:73}.T1.nii" "${i:0:73}.brain.nii" -m
done
