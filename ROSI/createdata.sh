#!\bin\bash

x=9
	while [ $x -lt 39 ]
	do	
		echo $x
		dir='/mnt/Data/DHCP' 
		#python ROSI/scriptSimulData.py --hr ../DHCP/image_$x.nii.gz --mask ../DHCP/binmask_$x.nii.gz --output ../simu/Large$x --name large$x --motion -8 8 &
  		x=$(( $x + 1 ))
	done

