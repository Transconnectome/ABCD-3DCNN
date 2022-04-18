#!bin/bash
#dir="/master_ssd/3DCNN/data/2.UKB"
dir="/storage/bigdata/UKB/raw/20252_UKB_T1"
destination="/master_ssd/3DCNN/data/2.UKB/1.sMRI"

count=1
add=1


for sub in $dir/*
do
        
    if [[ "${sub}" == *"_2_0"* ]]
    then
        sub_idx="${sub:38:7}" #subject number
        file="${sub:38}"
        #sub_idx="${sub:29:7}"
        #file="${sub:29}"
        #echo $file
        scp "${dir}/${file}" "${destination}"
        cd "${destination}"
        unzip "${destination}/${file}"
        rm "${destination}/${file}"
        mv "${destination}/T1/T1_brain.nii.gz" "${destination}"
        mv "${destination}/T1_brain.nii.gz" "${sub_idx}.nii.gz"
        rm -r "${destination}/T1"
        count=$(expr $count + $add)
        echo $count
    
        #progress bar
        #printf "\b${sp:i++%${#sp}:1}"
    fi
done

#pip3 install numpy 
#pip3 install nibabel
#echo `python3 "/master_ssd/3DCNN/data/2.UKB/UKB_data.py"`
#cd "${destination}"
#rm *.nii.gz
