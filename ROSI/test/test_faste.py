import faster_code.registration_all_algo as re
import joblib as job
import cProfile
import pstats
from pstats import SortKey
from numpy import array,zeros
results=job.load('../res/ipta5/sub-0034/ses-0043/res_test_dice.joblib.gz')
listSlice=results[0][1]
slice1=listSlice[12]
slice2=listSlice[60]
ge=results[3][1][0]
gn=results[4][1][0]
gi=results[5][1][0]
gu=results[6][1][0]
#re.costLocal(slice1,slice2)
#list_array=array(listSlice)
#cProfile.run('re.costLocal(slice1,slice1)','restats')
#p = pstats.Stats('restats')
#p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
#re.updateCostBetweenAllImageAndOne(12,listSlice,ge,gn,gi,gu)
#re.costLocal(slice1,slice2)
#cProfile.run('re.updateCostBetweenAllImageAndOne(12,list_array,ge,gn,gi,gu)','restats')
#p = pstats.Stats('restats')
#p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
x0=slice1.get_parameters()
i_slice=12
grid_slices=array([ge,gn,gi,gu])
set_o=zeros(len(listSlice))
lamb=10
Vmx=0
#re.cost_fct(x0,i_slice,listSlice,grid_slices,set_o,lamb,Vmx)
hyperparameters=[4,0.25,1e-10,2,10,4]
re.SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,i_slice,Vmx)
cProfile.run('re.SimplexOptimisation(x0,hyperparameters,listSlice,grid_slices,set_o,i_slice,Vmx)','restats')
p = pstats.Stats('restats')
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
print(len(listSlice))