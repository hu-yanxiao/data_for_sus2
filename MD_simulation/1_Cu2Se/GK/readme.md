step.1 
```bash
T=600 # 600K
for XXX in {random code};do;
mpirun -n N  /where/you/install/interface-lammps-mlip-2/lmp_intel_cpu_intelmpi -in lmp.in -var T $T -var random ${XXX} > lmp_${T}_${XXX}.out ;
done
```
step.2
```bash
python 3_hyx_heatflux2hcacf.py
```
step.3 
```bash
python python 2_plot_hcacf.py 600 --t1 20 --t2 100
```
