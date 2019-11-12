export OMP_NUM_THREADS
export GOMP_CPU_AFFINITY

ITE=$(seq 5) # nombre de mesures
THREADS="1 2 4 6 8 12" # nombre de threads à utiliser pour les expés
GOMP_CPU_AFFINITY=$(seq 0 11) # on fixe les threads

PARAM="./2Dcomp -n -k vie -i 100 -g 32 -a guns -s " # parametres commun à toutes les executions 

execute (){
EXE="$PARAM $*"
OUTPUT="$(echo $* | tr -d ' ')".data
for nb in $ITE; do for OMP_NUM_THREADS in $THREADS ; do echo -n "$OMP_NUM_THREADS " >> $OUTPUT  ; $EXE 2>> $OUTPUT ; done; done
}

# on suppose avoir codé 2 fonctions :
#   mandel_compute_omp_dynamic()
#   mandel_compute_omp_static()

for i in 512 4096 ;  # 2 tailles : -s 256 puis -s 512 
do
    execute $i -v seq_base
    execute $i -v omp_base_static
    execute $i -v omp_base_cylic
    execute $i -v omp_base_dynamic
    execute $i -v omp_base_collapse
done
