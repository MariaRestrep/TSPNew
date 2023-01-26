# This script generate the scripts to run all the instances on server grids.

# SBATCH --mail-user=simon.thevenin@imt-atlantique.fr
import sys
import csv


def CreatHeader(file, filename):
    CreatHeaderNantes(file, filename)


def CreatHeaderNantes(file, filename):
    file.write("""#!/bin/bash
# Nom du job
#SBATCH -J MON_JOB_MPI
#
# Partition visee
#SBATCH --partition=MPI-short
#
# Nombre de noeuds
#SBATCH --nodes=1
# Nombre de processus MPI par noeud
#SBATCH --ntasks-per-node=1
#SBATCH --mem 10000
#
# Temps de presence du job
#SBATCH --time=2:00:00
#
# Adresse mel de l'utilisateur
#
# Envoi des mails
#SBATCH --mail-type=abort,end
#
#SBATCH -o /home/LS2N/thevenin-s/log/job_mpi-%s.out

module purge
module load intel/2016.3.210
module load intel/mkl/64/2016.3.210
module load intel/mpi/2016.3.210
module load python/3.7.4
module load intel/mkl/64/2017.4.196
module load gcc/12.1.0


export LD_PRELOAD=/lib64/psm2-compat/libpsm_infinipath.so.1

#
# Faire le lien entre SLURM et Intel MPI
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

""" % (filename[8:]))


# SBATCH --mail-user=simon.thevenin@imt-atlantique.fr

def CreateJob(nb_packages, distribution, version, shiftMin, shiftMax, shiftG):
    qsub_filename = "./Jobs/job_tsp_n_%s_%s_%s_%s_%s_%s" % (nb_packages, distribution, version, shiftMin, shiftMax, shiftG)
    qsub_file = open(qsub_filename, 'w+')
    CreatHeader(qsub_file, qsub_filename)
    qsub_file.write("""
srun python main.py %s %s %s --full_patterns -t 7200  -minsl %s -maxsl %s -shiftgap %s > /home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt 
""" % (nb_packages, distribution, version, shiftMin, shiftMax, shiftG))

    return qsub_filename


if __name__ == "__main__":

    Packages = [50, 100, 200]

    Distribution = ["-u", "-n"]

    ShiftConf = [[6, 10, 1], [6, 10, 2], [6, 10, 4], [4, 8, 2], [8, 8, 2]]

    filenewname = "runtestOpt.sh"
    filenew = open(filenewname, 'w')
    filenew.write("""
    #!/bin/bash -l
    #
    """)

    for p in Packages:
        for d in Distribution:
            if p == 200 and d == "-n":
                continue
            else:
                for v in range(10):
                    index = 0
                    for shiftC in ShiftConf:
                        if p == 100 and index > 0:
                            jobname = CreateJob(p, d, v, shiftC[0], shiftC[1], shiftC[2])
                            filenew.write("sbatch %s \n" % (jobname))
                        elif p == 200 and index > 1:
                            jobname = CreateJob(p, d, v, shiftC[0], shiftC[1], shiftC[2])
                            filenew.write("sbatch %s \n" % (jobname))
                        elif p == 50:
                            jobname = CreateJob(p, d, v, shiftC[0], shiftC[1], shiftC[2])
                            filenew.write("sbatch %s \n" % (jobname))
                        index += 1
