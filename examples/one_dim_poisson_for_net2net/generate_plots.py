import os
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import pandas as pd
from examples.one_dim_poisson_for_net2net.train_student_after_net2net import (
    get_best_teacher_add,
    TEACHER_DIR,
    STUDENT_DIR,
    PROBLEM_DATA_DIR
)

num_seeds = 50
# TRAIN_TEACHER_DATA_DIR = pjoin(SCRIPT_DIR, f".tmp/teachers/")
# TRAIN_STUDENT_DATA_DIR = pjoin(SCRIPT_DIR, f".tmp/student/")
# TRUE_DATA_ADD = pjoin(SCRIPT_DIR, f"data/pde_data.csv")
TRAIN_TERMS_FOR_PLOT = ['bc','pde','compat','acc']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)

        


def plot_solution_teachers(best_teacher_id):
    plt.rcParams["figure.figsize"] = (8,6)

    transp = np.linspace(0., 1., int(num_seeds*3)+1)

    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(TEACHER_DIR, f"solution_before_train_{i}.csv")
        )
        plt.plot(data['x'], data['u'], linestyle='--', color='b', alpha=transp[3*i+1])
    data = pd.read_csv(pjoin(TEACHER_DIR, f"solution_before_train_{best_teacher_id}.csv"))
    plt.plot(data['x'], data['u'], linestyle='-', linewidth=4, color='r', alpha=transp[3*best_teacher_id+1])
    plt.ylim(-2.3, 2.3)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.tight_layout()
    plt.savefig(pjoin(TEACHER_DIR, 'u_before_training.png'))
    # plt.show()
    plt.close()

    
    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(TEACHER_DIR, f"solution_after_train_{i}.csv")
        )
        plt.plot(data['x'], data['u'], linestyle='--', color='b', alpha=transp[3*i+1])
    data = pd.read_csv(pjoin(TEACHER_DIR, f"solution_after_train_{best_teacher_id}.csv"))
    plt.plot(data['x'], data['u'], linestyle='-', linewidth=4, color='r', alpha=transp[3*best_teacher_id+1])
    plt.ylim(-2.3, 2.3)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.tight_layout()
    plt.savefig(pjoin(TEACHER_DIR, 'u_after_training.png'))
    # plt.show()
    plt.close()


    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(TEACHER_DIR, f"solution_before_train_{i}.csv")
        )
        plt.plot(data['x'], data['du'], linestyle='--', color='b', alpha=transp[3*i+1])
    data = pd.read_csv(pjoin(TEACHER_DIR, f"solution_before_train_{best_teacher_id}.csv"))
    plt.plot(data['x'], data['du'], linestyle='-', linewidth=4, color='r', alpha=transp[3*best_teacher_id+1])
    plt.ylim(-6., 6.)
    plt.xlabel('x')
    plt.ylabel('du')
    plt.tight_layout()
    plt.savefig(pjoin(TEACHER_DIR, 'du_before_training.png'))
    # plt.show()
    plt.close()


    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(TEACHER_DIR, f"solution_after_train_{i}.csv")
        )
        plt.plot(data['x'], data['du'], linestyle='--', color='b', alpha=transp[3*i+1])
    data = pd.read_csv(pjoin(TEACHER_DIR, f"solution_after_train_{best_teacher_id}.csv"))
    plt.plot(data['x'], data['du'], linestyle='-', linewidth=4, color='r', alpha=transp[3*best_teacher_id+1])
    plt.ylim(-6., 6.)
    plt.xlabel('x')
    plt.ylabel('du')
    plt.tight_layout()
    plt.savefig(pjoin(TEACHER_DIR, 'du_after_training.png'))
    # plt.show()
    plt.close()

def plot_error_student(best_teacher_id):
    plt.rcParams["figure.figsize"] = (8,6)


    data_best_techer = pd.read_csv(
        pjoin(TEACHER_DIR, f"solution_after_train_{best_teacher_id}.csv")
    )
    data_student = pd.read_csv(
        pjoin(STUDENT_DIR, f"solution_after_train_rand.csv")
    )
    PROBLEM_DATA_DIR
    data_true = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "pde_data.csv"))
    # plt.rcParams['text.usetex'] = True
    # plt.plot(data_true['x'], data_true['u'] - data_best_techer['u'], linestyle='-.', color='k', label='exact')
    plt.plot(data_best_techer['x'], np.abs(data_true['u'] - data_best_techer['u']), linestyle='--', color='r', label='teacher')
    plt.plot(data_student['x'], np.abs(data_true['u'] - data_student['u']), linestyle='--', color='b', label='student')
    plt.xlabel('x')
    plt.ylabel(r'|$u_{exact} - u_{pred}|$')
    plt.tight_layout()
    plt.legend()
    plt.savefig(STUDENT_DIR + 'u_teach_stud_exact.png')
    plt.show()
    plt.close()

    # plt.plot(data_true['x'], data_true['du'], linestyle='-.', color='k', label='exact')
    plt.plot(data_best_techer['x'], np.abs(data_true['du'] - data_best_techer['du']), linestyle='--', color='r', label='teacher')
    plt.plot(data_student['x'], np.abs(data_true['du'] - data_student['du']), linestyle='--', color='b', label='student')
    plt.xlabel('x')
    plt.ylabel(r'$|du_{exact} - du_{pred}|$')
    plt.tight_layout()
    plt.legend()
    plt.savefig(STUDENT_DIR + 'du_teach_stud_exact.png')
    plt.show()
    plt.close()


def plot_loss(
    data_df:pd.DataFrame, 
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    for case in TRAIN_TERMS_FOR_PLOT:
        plt.plot(data_df[case].to_numpy(), label=case)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    # plt.show()
    plt.close()

def plot_loss_relative(
    data_df_student:pd.DataFrame, 
    data_df_teacher:pd.DataFrame,
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    base_data_series = data_df_teacher.loc[len(data_df_teacher)-1]
    for case in TRAIN_TERMS_FOR_PLOT:
        plt.plot(data_df_student[case].to_numpy()/base_data_series[case], label=case)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('relative MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    plt.show()
    plt.close()

def plot_loss_relative_zoomin(
    data_df_student:pd.DataFrame, 
    data_df_teacher:pd.DataFrame,
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    base_data_series = data_df_teacher.loc[len(data_df_teacher)-1]
    for case in TRAIN_TERMS_FOR_PLOT:
        plt.plot(data_df_student[case].to_numpy()[:10]/base_data_series[case], label=case)
    plt.ylim(0.4, 2.)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('relative MSE')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    plt.show()
    plt.close()

if __name__ == "__main__":
    teacher_loss_add, teacher_score = get_best_teacher_add(TEACHER_DIR)
    teacher_id = int((teacher_loss_add.split("_")[-1]).split(".")[0])
    print(f"best teacher id {teacher_id}")
    plot_solution_teachers(teacher_id)
    plot_error_student(teacher_id)

    best_teacher_loss_df = pd.read_csv(pjoin(TEACHER_DIR, f"loss_train_{teacher_id}.csv"))
    plot_loss(best_teacher_loss_df, pjoin(TEACHER_DIR, f"loss_train_{teacher_id}.png"))
    student_loss_df = pd.read_csv(pjoin(STUDENT_DIR, f"loss_train_rand.csv"))
    plot_loss_relative(student_loss_df, best_teacher_loss_df,
        pjoin(STUDENT_DIR, f"loss_train_relative.png")
    )
    plot_loss_relative_zoomin(student_loss_df, best_teacher_loss_df,
        pjoin(STUDENT_DIR, f"loss_train_relative_zoomin.png")
    )