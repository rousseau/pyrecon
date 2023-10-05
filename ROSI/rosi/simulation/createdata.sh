#!\bin\bash

x=1
	while [ $x -lt 5 ]
	do	
		ecbo "image "${i} 
 		python scriptSimulData.py --hr ../DHCP/image_$x.nii.gz --mask ../DHCP/binmask_$x.nii.gz --output ../simu/tres_petit$x --name trespetit$x --motion -1 1
		echo "tres_petit created"
		python scriptSimulData.py --hr ../DHCP/image_$x.nii.gz --mask ../DHCP/binmask_$x.nii.gz --output ../simu/Petit$x --name petit$x --motion -3 3
		echo "Petit created"
		python scriptSimulData.py --hr ../DHCP/image_$x.nii.gz --mask ../DHCP/binmask_$x.nii.gz --output ../simu/Moyen$x --name moyen$x --motion -5 5
		echo "Grand created "
		python scriptSimulData.py --hr ../DHCP/image_$x.nii.gz --mask ../DHCP/binmask_$x.nii.gz --output ../simu/Grand$x --name grand$x --motion -8 8
  		x=$(( $x + 1 ))
	done

